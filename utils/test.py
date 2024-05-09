import csv

def find_missing_images(txt_file, csv_file, output_file):
  """
  Reads lines from a txt file, searches for them in the "Image" column of a CSV file, 
  and writes the not found lines to a new CSV file.

  Args:
      txt_file: Path to the text file containing image names (one per line).
      csv_file: Path to the CSV file containing an "Image" column.
      output_file: Path to the output CSV file for missing images.
  """
  missing_images = []
  found_images = set()

  # Read image names from txt file
  with open(txt_file, 'r') as f:
    for line in f:
      image_name = line.strip()
      missing_images.append(image_name+'.npz')

  # Read CSV file and check for image names
  with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader, "None")  # Skip header row
    for row in reader:
      image_column = 1 # Assuming "Image" is the column name
      if row[image_column] in missing_images:
        found_images.add(row[image_column])
        missing_images.remove(row[image_column])

  # Remove found images from missing list
  missing_images = [img for img in missing_images if img not in found_images]

  # Write missing images to output CSV
  with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(["Missing Image"])  # Write header row
    for image in missing_images:
      writer.writerow([image])

# Example usage
txt_file = "/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/train.txt"
csv_file = "/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/synapse_npz_200.csv"
output_file = "/home/sakir-w4-linux/Development/Thesis/CIBM/Datasets/Synapse/Ariadne/Train Set/missing_images.csv"
find_missing_images(txt_file, csv_file, output_file)

print("Missing images written to", output_file)