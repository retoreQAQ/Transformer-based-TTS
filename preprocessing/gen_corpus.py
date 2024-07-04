def process_csv(input_csv_path, output_path, output_with_id_path):
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        lines = csvfile.readlines()
        
        with open(output_path, 'w', encoding='utf-8') as txtfile, open(output_with_id_path, 'w', encoding='utf-8') as txtfile_with_id:
            for line in lines:
                # Split each line by '|'
                parts = line.strip().split('|')
                
                if len(parts) >= 3:
                    content = '|'.join(parts[2:])  # Join all elements after the second '|' as content
                    content = content.lower()  # Convert to lowercase
                    txtfile.write(content + '\n')
                    txtfile_with_id.write(parts[0] + ' ' + content + '\n')
                else:
                    print(f'Warning: Skipping row with fewer than 3 columns: {line}')

input_csv_path = '/home/you/workspace/database/LJSpeech-1.1/metadata.csv'
output_path = '/home/you/workspace/son/transformer-based-TTS/preprocessing/corpus.txt'
output_with_id_path = '/home/you/workspace/son/transformer-based-TTS/preprocessing/corpus_with_id.txt'
process_csv(input_csv_path, output_path, output_with_id_path)
