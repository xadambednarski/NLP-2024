import random
SEED = 2151839213825


if __name__ == "__main__":
    random.seed(SEED)
    file = open("../data/input/task_6-3_all.txt", "r")
    all_text = file.read().split("\n")
    samples_ids = random.sample(range(0, len(all_text)), 100)
    sample_file = open("../data/input/task_6-3_sample.txt", "w")
    for i in samples_ids:
        sample_file.write(all_text[i] + "\n")
    sample_file.close()
    file.close()
