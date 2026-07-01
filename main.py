import os
import logging
import random
from src.utils import load_yaml, setup_logger, save_csv
from src.preprocess import Preprocessor
from src.fraud_graph import FraudGraphBuilder
from src.faci import FACICalculator
from src.hybrid_optimizer import HybridOptimizer
from src.augmentor import Augmentor
from src.policy_memory import PolicyMemory
from src.replay_buffer import ReplayBuffer
from src.train_classifier import IncrementalClassifier
from src.evaluate import Evaluator
from src.visualizer import Visualizer

def main():
    config = load_yaml("configs/config.yaml")
    opt_config = load_yaml("configs/optimizer.yaml")
    data_config = load_yaml("configs/dataset.yaml")
    
    logger = setup_logger("HybridPipeline", config['paths']['outputs_dir'] + "/logs")
    logger.info("Initializing Continual Adaptive GA-GWO Pipeline...")
    
    preprocessor = Preprocessor(data_config)
    graph_builder = FraudGraphBuilder()
    faci_calc = FACICalculator()
    
    bounds = [
        opt_config['bounds']['bt_ratio'],
        opt_config['bounds']['bert_ratio'],
        opt_config['bounds']['budget'],
        opt_config['bounds']['mask_prob'],
        opt_config['bounds']['semantic_threshold'],
        opt_config['bounds']['entity_weight'],
        opt_config['bounds']['chunk_priority'],
        opt_config['bounds']['learning_rate']
    ]
    
    policy_memory = PolicyMemory(capacity=50, csv_path=os.path.join(config['paths']['results_dir'], "policy_memory.csv"))
    optimizer = HybridOptimizer(opt_config['genetic_algorithm'], opt_config['grey_wolf_optimization'], bounds, memory=policy_memory)
    augmentor = Augmentor()
    
    replay_buffer = ReplayBuffer(capacity=1000)
    classifier = IncrementalClassifier(num_classes=data_config['num_classes'])
    evaluator = Evaluator()
    visualizer = Visualizer(out_dir=config['paths']['visualizations_dir'])
    
    def get_chunk(chunk_id, size=10):
        texts = [
            f"My card was stolen and used for a scam of 500 dollars. I never gave them my OTP." if i % 2 == 0 
            else f"Fraudulent charge on my account for a very large amount. I suspect phishing breach."
            for i in range(size)
        ]
        return texts, [random.randint(0, data_config['num_classes']-1) for _ in range(size)]
    
    num_chunks = config['runtime']['total_samples'] // config['runtime']['chunk_size']
    
    training_loss_history = []
    faci_history = []
    ga_div_global = []
    ga_mut_global = []
    
    for chunk_id in range(num_chunks):
        logger.info(f"Processing Chunk {chunk_id + 1}/{num_chunks}")
        texts, labels = get_chunk(chunk_id, config['runtime']['chunk_size'])
        
        chunk_faci_scalars = []
        processed_data = []
        
        last_faci_dict = None
        for text, label in zip(texts, labels):
            proc = preprocessor.process(text)
            G = graph_builder.build(proc['entities'])
            density = graph_builder.get_entity_density(G, len(proc['processed_text'].split()))
            faci_dict = faci_calc.compute(proc['processed_text'], density)
            
            chunk_faci_scalars.append(faci_dict["scalar"])
            last_faci_dict = faci_dict
            processed_data.append({"text": proc['processed_text'], "entities": proc['entities'], "label": label})
            
        avg_faci_scalar = sum(chunk_faci_scalars) / max(1, len(chunk_faci_scalars))
        faci_history.append(avg_faci_scalar)
        
        opt_result = optimizer.optimize(chunk_id, last_faci_dict)
        alpha_policy = opt_result['policy']
        
        # Adaptive Budget Override based on multi-dim FACI
        alpha_policy[2] = last_faci_dict["recommended_budget"]
        logger.info(f"Selected Policy (Budget Adapted): {alpha_policy}")
        
        ga_div_global.extend(opt_result['ga_diversity'])
        ga_mut_global.extend(opt_result['ga_mutation'])
        
        augmented_batch = []
        for p_data in processed_data:
            augs = augmentor.generate(p_data['text'], p_data['entities'], alpha_policy)
            for a in augs:
                augmented_batch.append((a, p_data['label']))
            augmented_batch.append((p_data['text'], p_data['label']))
            
        replay_buffer.add(augmented_batch)
        train_batch = replay_buffer.sample(batch_size=32)
        
        lr = alpha_policy[7]
        loss = classifier.train_on_batch(train_batch, learning_rate=lr)
        training_loss_history.append(loss)
        
        eval_metrics = evaluator.evaluate(classifier.model, classifier.tokenizer, train_batch, classifier.device)
        
        policy_memory.add_state(
            chunk_id=chunk_id, 
            policy=alpha_policy.tolist(), 
            metrics=eval_metrics,
            alpha=opt_result['alpha'],
            beta=opt_result['beta'],
            delta=opt_result['delta'],
            elite_pop=opt_result['elite_policies']
        )
        
        csv_path = os.path.join(config['paths']['results_dir'], "optimizer_history", "history.csv")
        sample_text_used = texts[0] if texts else "Empty Chunk"
        
        save_csv([chunk_id, sample_text_used, avg_faci_scalar, opt_result['fitness']] + alpha_policy.tolist(), 
                 csv_path, 
                 headers=["Chunk", "Sample_Text", "FACI", "Fitness", "BT", "BERT", "Budget", "Mask", "Sem", "Ent", "Pri", "LR"])
        
        if chunk_id % 5 == 0 or chunk_id == num_chunks - 1:
            visualizer.plot_training_metrics(training_loss_history)
            visualizer.plot_faci_distribution(faci_history)
            visualizer.plot_faci_radar(last_faci_dict)
            visualizer.plot_optimizer_convergence(opt_result['ga_history'], opt_result['gwo_history'], filename=f"convergence_chunk_{chunk_id}.png")
            visualizer.plot_ga_metrics(ga_div_global, ga_mut_global)
            
    logger.info("Pipeline Complete!")

if __name__ == "__main__":
    main()
