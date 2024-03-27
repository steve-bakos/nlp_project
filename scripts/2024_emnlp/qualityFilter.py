import os, shutil
import argparse
from comet import download_model, load_from_checkpoint


if __name__ == "__main__":
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    threshold = args.threshold #range -> 0 to 1, increase value to filter out more.

    # Open the file for reading
    with open(args.input_file, 'r') as file:
        # Loop through each line in the file
        data = []
        for line in file:
            data.append({"src": line.strip().split('|||')[0], "mt": line.strip().split('|||')[1]})

    model_output = model.predict(data, batch_size=args.batch_size, gpus=1)

    line_counter = 0
    with open(args.output_file, 'w', encoding='utf-8') as file:
        for i in range (len(data)):
            if model_output[0][i] > threshold:
                line = f"{data[i]['src']} ||| {data[i]['mt']}\n"
                file.write(line)
                line_counter += 1

    print("filtered out ", len(data) - line_counter, "pairs out of ", len(data), "for file ", args.input_file)
                    

                     
                     
                     