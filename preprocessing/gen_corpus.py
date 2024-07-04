import csv

def process_csv(input_csv_path, output_path, output_with_id_path):
    # Open the input CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        
        # Open the output TXT file
        with open(output_path, 'w', encoding='utf-8') as txtfile, open(output_with_id_path, 'w', encoding='utf-8') as txtfile_with_id:
            for row in reader:
                if len(row) > 2:
                    # Get the content after the second '|' delimiter and convert to lowercase
                    content = row[2].lower()
                    txtfile.write(content + '\n')
                    txtfile_with_id.write(row[0] + ' ' + content + '\n')

input_csv_path = '/home/you/workspace/son/database/LJSpeech-1.1/metadata.csv'
output_path = '/home/you/workspace/son/transformer-based-TTS/preprocessing/corpus.txt'
output_with_id_path = '/home/you/workspace/son/transformer-based-TTS/preprocessing/corpus_with_id.txt'
process_csv(input_csv_path, output_path, output_with_id_path)
