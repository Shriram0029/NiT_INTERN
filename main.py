import os
import json
import logging
import random
import numpy as np
import pandas as pd
from collections import Counter

from src.utils import load_yaml, setup_logger, save_csv
from src.preprocess import Preprocessor
from src.fraud_graph import FraudGraphBuilder
from src.faci import FACICalculator
from src.class_imbalance import ClassImbalanceAnalyzer
from src.hybrid_optimizer import HybridOptimizer
from src.augmentor import Augmentor
from src.policy_memory import PolicyMemory
from src.replay_buffer import ReplayBuffer
from src.train_classifier import IncrementalClassifier
from src.evaluate import Evaluator
from src.visualizer import Visualizer
from src.explainability import ExplainabilityEngine
from src.semantic_validator import SemanticValidator
from src.statistical_analysis import StatisticalAnalyzer

def main():
    config = load_yaml("configs/config.yaml")
    opt_config = load_yaml("configs/optimizer.yaml")
    data_config = load_yaml("configs/dataset.yaml")
    
    logger = setup_logger("HybridPipeline", config['paths']['outputs_dir'] + "/logs")
    logger.info("Initializing Master Adaptive GA-GWO Pipeline...")
    
    # 1. Dataset Pipeline
    preprocessor = Preprocessor(data_config)
    data_path = os.path.join("data", "complaints_150.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    filtered_data = preprocessor.filter_and_audit(raw_data, config['paths']['outputs_dir'])
    
    with open(os.path.join("data", "filtered_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4)
        
    # Map top 5 products to 0-4
    product_counts = Counter(item.get("product", "") for item in filtered_data)
    top_5_products = [p for p, c in product_counts.most_common(5)]
    product_to_label = {p: i for i, p in enumerate(top_5_products)}
    
    clean_data = []
    for item in filtered_data:
        text = item.get("complaint_what_happened", "").strip()
        product = item.get("product", "")
        if product in product_to_label:
            clean_data.append((text, product_to_label[product]))
            
    num_chunks = config['runtime']['total_samples'] // config['runtime']['chunk_size']
    required_samples = num_chunks * config['runtime']['chunk_size'] + 50
    while len(clean_data) < required_samples:
        clean_data.extend(clean_data)
    random.shuffle(clean_data)
    
    train_data = clean_data[:-50]
    test_data = clean_data[-50:]
    
    def get_chunk(chunk_id, size=10, is_test=False):
        if is_test:
            return [item[0] for item in test_data[:size]], [item[1] for item in test_data[:size]]
        start_idx = (chunk_id * size) % len(train_data)
        chunk_items = train_data[start_idx : start_idx + size]
        return [item[0] for item in chunk_items], [item[1] for item in chunk_items]
        
    # 3. Class Imbalance Analyzer
    imbalance_analyzer = ClassImbalanceAnalyzer()
    imbalance_analyzer.fit([item[1] for item in train_data])
    
    # 17. Statistical Analysis
    stat_analyzer = StatisticalAnalyzer()
    
    baseline_modes = config['runtime'].get('baseline_modes', ['hybrid'])
    
    for baseline in baseline_modes:
        logger.info(f"============================================================")
        logger.info(f"Running Experiment Baseline: {baseline}")
        logger.info(f"============================================================")
        
        graph_builder = FraudGraphBuilder()
        faci_calc = FACICalculator()
        augmentor = Augmentor()
        evaluator = Evaluator()
        visualizer = Visualizer(out_dir=os.path.join(config['paths']['visualizations_dir'], baseline))
        explain_engine = ExplainabilityEngine()
        
        bounds = [
            opt_config['bounds']['bt_ratio'], opt_config['bounds']['bert_ratio'],
            opt_config['bounds']['budget'], opt_config['bounds']['mask_prob'],
            opt_config['bounds']['semantic_threshold'], opt_config['bounds']['entity_weight'],
            opt_config['bounds']['chunk_priority'], opt_config['bounds']['learning_rate']
        ]
        
        baseline_results_dir = os.path.join(config['paths']['results_dir'], baseline)
        os.makedirs(baseline_results_dir, exist_ok=True)
        
        mem_path = os.path.join(baseline_results_dir, "policy_memory.csv")
        policy_memory = PolicyMemory(capacity=50, csv_path=mem_path)
        optimizer = HybridOptimizer(opt_config['genetic_algorithm'], opt_config['grey_wolf_optimization'], bounds, memory=policy_memory)
        
        replay_buffer = ReplayBuffer(capacity=1000, num_classes=data_config['num_classes'])
        classifier = IncrementalClassifier(num_classes=data_config['num_classes'])
        
        training_loss_history = []
        faci_history = []
        strategy_history = []
        faci_data_list = []
        
        val_report_path = os.path.join(baseline_results_dir, "validation_report.csv")
        with open(val_report_path, 'w') as f:
            f.write("Chunk,Original,Augmented,Accepted,Reason\n")

            
        test_f1_scores = []
        
        for chunk_id in range(num_chunks):
            logger.info(f"--- Chunk {chunk_id + 1}/{num_chunks} ---")
            texts, labels = get_chunk(chunk_id, config['runtime']['chunk_size'])
            
            processed_data = []
            chunk_faci_dicts = []
            
            for text, label in zip(texts, labels):
                proc = preprocessor.process(text)
                if proc is None: continue
                
                G = graph_builder.build(proc['entities'])
                density = graph_builder.get_entity_density(G, len(proc['processed_text'].split()))
                
                cif = imbalance_analyzer.get_factor(label)
                faci_dict = faci_calc.compute(proc['processed_text'], density, class_imbalance_factor=cif)
                chunk_faci_dicts.append(faci_dict)
                faci_data_list.append(faci_dict)
                processed_data.append({"text": proc['processed_text'], "entities": proc['entities'], "label": label})
                
            if not processed_data: continue
            
            # Use average FACI vector for the chunk
            avg_faci_dict = {k: np.mean([fd[k] for fd in chunk_faci_dicts]) for k in chunk_faci_dicts[0].keys()}
            avg_faci_dict['recommended_budget'] = int(np.mean([fd['recommended_budget'] for fd in chunk_faci_dicts]))
            faci_history.append(avg_faci_dict['scalar'])
            
            if baseline == 'hybrid':
                opt_result = optimizer.optimize(chunk_id, avg_faci_dict)
                alpha_policy = opt_result['policy']
                prediction = opt_result['prediction']
            else:
                # Mock baseline policies
                alpha_policy = np.zeros(8)
                alpha_policy[2] = avg_faci_dict["recommended_budget"]
                alpha_policy[4], alpha_policy[5], alpha_policy[7] = 0.8, 0.8, 2e-5
                
                if baseline == 'no_aug': alpha_policy[2] = 0
                elif baseline == 'fixed_bt': alpha_policy[0] = 1.0
                elif baseline == 'fixed_bert': alpha_policy[1] = 1.0
                elif baseline == 'random': alpha_policy = np.array([random.random() for _ in range(8)])
                elif baseline == 'rule_based':
                    if avg_faci_dict['scalar'] < 0.3: alpha_policy[2] = 0
                    elif avg_faci_dict['scalar'] <= 0.6: alpha_policy[1] = 1.0
                    else: alpha_policy[0] = 1.0; alpha_policy[1] = 1.0
                elif baseline == 'ga_only':
                    opt_result = optimizer.optimize(chunk_id, avg_faci_dict)
                    alpha_policy = opt_result['elite_policies'][0]
                elif baseline == 'gwo_only':
                    opt_result = optimizer.optimize(chunk_id, avg_faci_dict)
                    alpha_policy = opt_result['alpha']
                    
                prediction = {
                    "strategy": optimizer._determine_strategy(alpha_policy[0], alpha_policy[1]),
                    "budget": alpha_policy[2],
                    "expected_utility": 0.0, "expected_cost": 0.0, "expected_macro_f1": 0.0,
                    "confidence": 0.0, "reason": f"Baseline {baseline} applied."
                }
                opt_result = {"alpha": alpha_policy, "beta": alpha_policy, "delta": alpha_policy, "elite_policies": [alpha_policy.tolist()]}
                
            strategy_history.append(prediction['strategy'])
            
            # Explainability
            explanation = explain_engine.generate_explanation(chunk_id, avg_faci_dict, prediction)
            logger.info(explanation)
            
            opt_hist_dir = os.path.join(baseline_results_dir, "optimizer_history")
            os.makedirs(opt_hist_dir, exist_ok=True)
            csv_path = os.path.join(opt_hist_dir, "history.csv")
            sample_text_used = texts[0] if texts else "Empty Chunk"
            save_csv([chunk_id, sample_text_used, avg_faci_dict['scalar'], opt_result.get('fitness', 0.0)] + alpha_policy.tolist(), 
                     csv_path, 
                     headers=["Chunk", "Sample_Text", "FACI", "Fitness", "BT", "BERT", "Budget", "Mask", "Sem", "Ent", "Pri", "LR"])
            
            # Semantic Validation & Generation
            semantic_validator = SemanticValidator(semantic_threshold=alpha_policy[4], entity_weight=alpha_policy[5])
            augmented_batch = []
            
            with open(val_report_path, 'a') as f:
                for p_data in processed_data:
                    existing = [t for t, _ in augmented_batch]
                    augs = augmentor.generate(p_data['text'], p_data['entities'], alpha_policy)
                    
                    for a in augs:
                        is_valid, reject_reason = semantic_validator.validate(p_data['text'], a, p_data['label'], p_data['label'], p_data['entities'], existing)
                        f.write(f"{chunk_id},\"{p_data['text']}\",\"{a}\",{is_valid},\"{reject_reason}\"\n")
                        if is_valid:
                            augmented_batch.append((a, p_data['label']))
                            
                    augmented_batch.append((p_data['text'], p_data['label']))
                    
            replay_buffer.add(augmented_batch)
            train_batch = replay_buffer.sample(batch_size=32)
            
            # Training
            lr = alpha_policy[7]
            loss = classifier.train_on_batch(train_batch, learning_rate=lr)
            training_loss_history.append(loss)
            
            # Evaluation
            test_batch = [(t, l) for t, l in zip(*get_chunk(0, size=50, is_test=True))]
            eval_metrics = evaluator.evaluate(classifier.model, classifier.tokenizer, test_batch, classifier.device)
            test_f1_scores.append(eval_metrics['macro_f1'])
            
            # Feedback Loop (Memory update)
            eval_metrics['fitness'] = prediction.get('expected_utility', eval_metrics['macro_f1'])
            eval_metrics['semantic_preservation'] = alpha_policy[4]
            eval_metrics['entity_preservation'] = alpha_policy[5]
            eval_metrics['cost'] = prediction.get('expected_cost', 0)
            
            policy_memory.add_state(
                chunk_id=chunk_id, policy=alpha_policy.tolist(), metrics=eval_metrics,
                alpha=opt_result['alpha'], beta=opt_result.get('beta', alpha_policy), delta=opt_result.get('delta', alpha_policy),
                elite_pop=opt_result.get('elite_policies', []), faci_dict=avg_faci_dict
            )
            
            # Visualization Logging
            if baseline == 'hybrid' and (chunk_id % 5 == 0 or chunk_id == num_chunks - 1):
                visualizer.plot_training_metrics({'loss': training_loss_history, 'test_macro_f1': test_f1_scores})
                visualizer.plot_faci_distribution(faci_history)
                visualizer.plot_faci_radar(avg_faci_dict)
                visualizer.plot_strategy_distribution(strategy_history)
                visualizer.plot_replay_distribution(replay_buffer.get_statistics()["class_distribution"])
                
                if 'ga_metrics' in opt_result:
                    visualizer.plot_ga_metrics(opt_result['ga_metrics'])
                if 'gwo_metrics' in opt_result:
                    visualizer.plot_gwo_metrics(opt_result['gwo_metrics'])
                    visualizer.plot_optimizer_convergence(opt_result['ga_metrics']['fitness_history'], opt_result['gwo_metrics']['alpha_fitness'])
                    
        # Post-baseline steps
        visualizer.plot_confusion_matrix(eval_metrics['confusion_matrix'], [f"Class {i}" for i in range(data_config['num_classes'])])
        visualizer.plot_faci_correlation(faci_data_list)
        
        stat_analyzer.add_experiment_results(baseline, test_f1_scores)
        
    # Statistical Analysis
    logger.info("Running Statistical Analysis...")
    stat_results = stat_analyzer.run_full_analysis(target_baseline="hybrid")
    
    stat_path = os.path.join(config['paths']['results_dir'], "statistical_analysis.json")
    with open(stat_path, "w") as f:
        json.dump(stat_results, f, indent=4)
        
    logger.info("Master Pipeline Complete!")

if __name__ == "__main__":
    main()

