import submitit
import os

def run_training():
    print("Inizio l'esecuzione di train.py sul cluster...")
    os.system("python train.py")
    print("Addestramento completato!")

if __name__ == "__main__":
    # La cartella dove verranno salvati i log di Slurm (stdout e stderr)
    log_dir = "slurm_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Inizializza l'Executor di submitit
    executor = submitit.AutoExecutor(folder=log_dir)
    
    # ==========================================
    # CONFIGURAZIONE SLURM
    # ==========================================
    # Modifica questi parametri in base alle policy del tuo cluster universitario.
    # Se il tuo cluster ha una partizione specifica per le GPU (es. 'gpu', 'cuda', 'dgx'), inseriscila.
    executor.update_parameters(
        slurm_partition="gpu",       # <-- Cambia con il nome della partizione GPU del tuo cluster
        gpus_per_node=1,             # Numero di GPU richieste
        tasks_per_node=1,            
        cpus_per_task=8,             # Numero di core CPU per preparare i batch veloci
        slurm_mem="32G",             # RAM richiesta
        timeout_min=60 * 24 * 2      # Tempo massimo (es. 48 ore)
    )
    
    print("Sottomissione del job a SLURM in corso...")
    job = executor.submit(run_training)
    
    print(f"✅ Job inviato con successo!")
    print(f"ID del Job: {job.job_id}")
    print(f"Puoi controllare lo stato con il comando: squeue -u $USER")
    print(f"I log verranno salvati in: {log_dir}/")
