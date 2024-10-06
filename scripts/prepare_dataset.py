import random
SEED = 3916111


if __name__ == "__main__":
    random.seed(SEED)
    file = open("task_6-1_all.txt", "r")
    all_text = file.read().split("\n")
    samples_ids = random.sample(range(0, len(all_text)), 100)
    sample_file = open("task_6-1_sample.txt", "w")
    for i in samples_ids:
        sample_file.write(all_text[i] + "\n")
    sample_file.close()
    file.close()
