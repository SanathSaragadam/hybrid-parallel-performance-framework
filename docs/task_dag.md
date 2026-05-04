# Task Flow / DAG

## Main Tasks

T0: Load PM100 / F-DATA dataset 
T1: Preprocess data and select useful numeric/system features 
T2: Partition dataset into chunks 
T3: Run NOTEARS on each chunk 
T4: Generate local DAGs 
T5: Aggregate and merge local DAGs 
T6: Filter weak relations 
T7: Generate global DAG and causal chain views 
T8: Run scaling experiments 
T9: Generate report and presentation visuals 

## Dependencies

T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9
