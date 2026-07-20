import re
import os

with open("main.py", "r", encoding="utf-8") as f:
    code = f.read()

# Target 1: seeds
code = re.sub(
    r'seeds = \[42, 123, 456\]',
    r'seeds = [42, 123, 456, 789, 101112]',
    code
)

# Target 6: reset_model and early stopping init
code = re.sub(
    r'(classifier\.reset_model\(\)\s+)(pm_csv = f"\{output_dir\}/policy_memory\.csv")',
    r'classifier.reset_model(total_steps=num_chunks * hyperparams["epochs"] * 2)\n\n'
    r'        best_es_f1 = -1.0\n'
    r'        patience_counter = 0\n'
    r'        es_patience = 3\n'
    r'        min_delta = 0.001\n'
    r'        best_chunk = -1\n'
    r'        best_model_path = f"{output_dir}/best_model_{seed}.pt"\n'
    r'        epochs_saved = 0\n\n'
    r'        \2',
    code
)

# Target 2: dynamic_threshold
code = re.sub(
    r'(avg_vector = np\.mean\(faci_vectors, axis=0\)\s+mh\["faci_scalar"\]\.append\(avg_scalar\))',
    r'\1\n\n'
    r'            if avg_scalar > 0.22:\n'
    r'                dynamic_threshold = 0.90\n'
    r'            elif avg_scalar >= 0.18:\n'
    r'                dynamic_threshold = 0.85\n'
    r'            else:\n'
    r'                dynamic_threshold = 0.80',
    code
)

# Target 3: pass dynamic_threshold
code = re.sub(
    r'augs      = augmentor\.generate\(text, label_idx, chunk_id, prediction\)',
    r'augs      = augmentor.generate(text, label_idx, chunk_id, prediction, dynamic_threshold=dynamic_threshold)',
    code
)

# Target 4: train_on_batch and evaluation logging
old_train = r'''            loss = classifier.train_on_batch\(
                train_batch, 
                learning_rate=hyperparams\["learning_rate"\], 
                epochs_per_chunk=hyperparams\["epochs"\]
            \)
            mh\["loss"\]\.append\(loss\)

            print\(f"\[\{baseline\}\]\[Chunk \{chunk_id\}\] Evaluating\.\.\.", flush=True\)
            eval_res = evaluator\.evaluate\(
                classifier\.model, classifier\.tokenizer, test_batch, classifier\.device
            \)
            f1 = eval_res\.get\("macro_f1", 0\.0\)
            mh\["macro_f1"\]\.append\(f1\)'''

new_train = r'''            loss, loss_fn_name, class_weights, gamma, last_lr = classifier.train_on_batch(
                train_batch, 
                learning_rate=hyperparams["learning_rate"], 
                epochs_per_chunk=hyperparams["epochs"]
            )
            mh["loss"].append(loss)

            print(f"[{baseline}][Chunk {chunk_id}] Evaluating...", flush=True)
            eval_res = evaluator.evaluate(
                classifier.model, classifier.tokenizer, test_batch, classifier.device
            )
            f1 = eval_res.get("macro_f1", 0.0)
            val_loss = eval_res.get("val_loss", 0.0)
            mh["macro_f1"].append(f1)
            
            # Log to training_history.csv
            hist_path = f"{output_dir}/training_history.csv"
            write_header = not os.path.exists(hist_path)
            with open(hist_path, "a", newline="", encoding="utf-8") as hf:
                import csv
                hw = csv.writer(hf)
                if write_header:
                    hw.writerow(["Seed", "Baseline", "Chunk", "Epoch", "Step", "Current_LR", "Train_Loss", "Validation_Loss", "Macro_F1", "Loss_Function", "Class_Weights", "Gamma"])
                hw.writerow([seed, baseline, chunk_id, chunk_id, classifier.current_step, last_lr, loss, val_loss, f1, loss_fn_name, str(class_weights), gamma])'''

code = re.sub(old_train, new_train, code)

# Target 5: Policy Memory update
old_pm = r'''                    budget=prediction\.get\("budget", 0\),
                    alpha=alpha_policy,
                \)'''

new_pm = r'''                    budget=prediction.get("budget", 0),
                    alpha=alpha_policy,
                    semantic_threshold=dynamic_threshold
                )'''

code = re.sub(old_pm, new_pm, code)

# Target 6: Early stopping check
old_telemetry = r'''            # ── Timing & memory telemetry ──────────────────'''
new_es = r'''            # ── Early Stopping Check ───────────────────────
            if baseline == "Hybrid GA + GWO":
                if f1 > best_es_f1 + min_delta:
                    best_es_f1 = f1
                    best_chunk = chunk_id
                    patience_counter = 0
                    torch.save(classifier.model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= es_patience:
                    logger.info(f"Early stopping triggered at chunk {chunk_id}! Restoring best model from chunk {best_chunk}.")
                    epochs_saved = num_chunks - chunk_id - 1
                    break

            # ── Timing & memory telemetry ──────────────────'''

code = code.replace(old_telemetry, new_es)

# Target 7: Best model restore
old_report = r'''            # Populate final report data
            final_report_data\["faci_stats"\] = \{'''

new_restore = r'''            # Restore best model for final evaluation if available
            if os.path.exists(best_model_path):
                classifier.model.load_state_dict(torch.load(best_model_path))
                eval_res = evaluator.evaluate(
                    classifier.model, classifier.tokenizer, test_batch, classifier.device
                )
                
            final_report_data["early_stopping"] = {
                "best_epoch": best_chunk,
                "best_macro_f1": best_es_f1,
                "training_epochs_saved": epochs_saved
            }
            
            # Populate final report data
            final_report_data["faci_stats"] = {'''

code = re.sub(old_report, new_restore, code)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("main.py updated successfully!")
