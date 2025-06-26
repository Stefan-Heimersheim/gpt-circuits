# Default parameters
NUM_SAMPLES=2
NUM_PROMPTS=5
SEED=125

# Parameters to loop over
RUN_INDEX="share-noshare_5P"
EDGE_SELECTIONS=("gradient")
UPSTREAM_LAYERS=(3)
SAE_VARIANTS=("share" "noshare")
EDGE_SET=(1      2      4      5      7     11     16     22     32     45
     63     90    127    181    256    362    512    724   1024   1448
   2048   2896   4095   5792   8191  11585  16383  23170  32768  46340
  65536  92681 131072 185363 262144)

# Create log directory if it doesn't exist
LOG_DIR="xavier/experiments/logs"
mkdir -p $LOG_DIR

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")

echo "Starting experiments..."
echo "- Number of samples: $NUM_SAMPLES"
echo "- Number of prompts: $NUM_PROMPTS"
echo "- Edge selection strategies: ${EDGE_SELECTIONS[*]}"
echo "- SAE variants: ${SAE_VARIANTS[*]}"
echo "- Random seed: $SEED"
echo ""

# Data generation
for SAE_VARIANT in "${SAE_VARIANTS[@]}"
do
  echo "Running experiments with SAE variant: $SAE_VARIANT"
  
  for EDGE_SELECTION in "${EDGE_SELECTIONS[@]}"
  do
    echo "Running experiments with edge selection strategy: $EDGE_SELECTION"
    
    for CURRENT_LAYER in "${UPSTREAM_LAYERS[@]}"
    do
      echo "Running experiments for upstream layer: $CURRENT_LAYER"
      
      # Run for logarithmically spaced number of edges
      for NUM_EDGES in "${EDGE_SET[@]}"
      do
        echo "Running experiment with SAE variant $SAE_VARIANT, layer $CURRENT_LAYER, $NUM_EDGES edges, $EDGE_SELECTION strategy..."
        
        # Run the Python script with the specified parameters
        python xavier/experiments/ablation_magnitudes_trans_block.py \
          --num-edges $NUM_EDGES \
          --upstream-layer-num $CURRENT_LAYER \
          --num-samples $NUM_SAMPLES \
          --num-prompts $NUM_PROMPTS \
          --edge-selection $EDGE_SELECTION \
          --sae-variant $SAE_VARIANT \
          --run-index $RUN_INDEX \
          --seed $SEED \
          2>&1 | tee "${LOG_DIR}/sae_${SAE_VARIANT}_layer${CURRENT_LAYER}_${EDGE_SELECTION}_edges_${NUM_EDGES}_${TIMESTAMP}.log"
        
        echo "Completed experiment with SAE variant $SAE_VARIANT, layer $CURRENT_LAYER, $NUM_EDGES edges, $EDGE_SELECTION strategy"
        echo ""

      done
    done
  done
done


echo "All data generation completed!"