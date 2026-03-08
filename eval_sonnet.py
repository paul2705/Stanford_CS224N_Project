from evaluation import test_sonnet

# Uses the default paths
chrf_score = test_sonnet(
    test_path='predictions/generated_sonnets_dev.txt',
    gold_path='data/TRUE_sonnets_held_out_dev.txt'
)
print(f"CHRF Score: {chrf_score}")