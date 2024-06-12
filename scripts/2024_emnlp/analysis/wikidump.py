import requests
import bz2
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
import time
import fugashi
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Language map dictionary
language_map = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

base_url = "https://dumps.wikimedia.org"
dump_directory = './dumps/'  # Directory to store downloaded dumps
token_counts = defaultdict(int)  # Dictionary to store token counts for each language

# Ensure dump directory exists
os.makedirs(dump_directory, exist_ok=True)

# Function to get the appropriate tokenizer for each language
def get_tokenizer(lang_code):
    if lang_code == "zh":
        print("Using bert-base-chinese tokenizer")
        return AutoTokenizer.from_pretrained("bert-base-chinese")
    elif lang_code == "ja":
        print("Using cl-tohoku/bert-base-japanese tokenizer with unidic")
        tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        return tokenizer
    elif lang_code == "ko":
        print("Using klue/roberta-base tokenizer")
        return AutoTokenizer.from_pretrained("klue/roberta-base")
    elif lang_code == "ar":
        print("Using asafaya/bert-base-arabic tokenizer")
        return AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    elif lang_code == "he":
        print("Using onlplab/alephbert-base tokenizer")
        return AutoTokenizer.from_pretrained("onlplab/alephbert-base")
    elif lang_code == "ru":
        print("Using DeepPavlov/rubert-base-cased tokenizer")
        return AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    elif lang_code == "sv":
        print("Using KB/bert-base-swedish-cased tokenizer")
        return AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
    elif lang_code == "uk":
        print("Using Ukrainian-NLP/uk-bert-base tokenizer")
        return AutoTokenizer.from_pretrained("Ukrainian-NLP/uk-bert-base")
    elif lang_code == "vi":
        print("Using vinai/phobert-base tokenizer")
        return AutoTokenizer.from_pretrained("vinai/phobert-base")
    elif lang_code == "hi":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    elif lang_code == "th":
        print("Using thichlm/mt5-small-thai tokenizer")
        return AutoTokenizer.from_pretrained("thichlm/mt5-small-thai")
    elif lang_code == "fa":
        print("Using HooshvareLab/bert-fa-base-uncased tokenizer")
        return AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    elif lang_code == "fi":
        print("Using TurkuNLP/bert-base-finnish-cased-v1 tokenizer")
        return AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    elif lang_code == "hu":
        print("Using SZTAKI-HLT/hubert-base-cc tokenizer")
        return AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
    elif lang_code == "it":
        print("Using dbmdz/bert-base-italian-cased tokenizer")
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")
    elif lang_code == "pl":
        print("Using dkleczek/bert-base-polish-uncased-v1 tokenizer")
        return AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
    elif lang_code == "pt":
        print("Using neuralmind/bert-base-portuguese-cased tokenizer")
        return AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    elif lang_code == "ro":
        print("Using dumitrescustefan/bert-base-romanian-cased-v1 tokenizer")
        return AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    elif lang_code == "tr":
        print("Using dbmdz/bert-base-turkish-cased tokenizer")
        return AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    elif lang_code == "bn":
        print("Using sagorsarker/bangla-bert-base tokenizer")
        return AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
    elif lang_code == "ur":
        print("Using urduhack/bert-base-urdu tokenizer")
        return AutoTokenizer.from_pretrained("urduhack/bert-base-urdu")
    elif lang_code == "te":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    elif lang_code == "mr":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    elif lang_code == "ml":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    elif lang_code == "kn":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    elif lang_code == "ta":
        print("Using ai4bharat/indic-bert tokenizer")
        return AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    else:
        print(f"Using distilbert-base-multilingual-cased tokenizer for {lang_code}")
        return AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# Function to download a dump file with retry logic
def download_dump(language_code, retries=3, delay=5):
    url = f"{base_url}/{language_code}wiki/latest/{language_code}wiki-latest-pages-articles.xml.bz2"
    file_path = f"{dump_directory}/{language_code}wiki-latest-pages-articles.xml.bz2"
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping download.")
        return file_path
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                        desc=f"Downloading {language_code}",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024):
                        size = f.write(chunk)
                        bar.update(size)
                return file_path
            else:
                print(f"Failed to download {url} (status code {response.status_code})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
    print(f"Failed to download {url} after {retries} attempts")
    return None

# Function to decompress a .bz2 file
def decompress_bz2(file_path):
    decompressed_file_path = file_path.replace('.bz2', '')
    if os.path.exists(decompressed_file_path):
        print(f"{decompressed_file_path} already exists. Skipping decompression.")
        return decompressed_file_path
    with bz2.open(file_path, 'rb') as f_in, open(decompressed_file_path, 'wb') as f_out:
        for data in iter(lambda: f_in.read(100 * 1024), b''):
            f_out.write(data)
    return decompressed_file_path

# Function to count tokens using the tokenizer
def count_tokens(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

# Process a batch of chunks with a progress bar
def process_chunk_batch(chunk_batch, tokenizer, progress_bar):
    batch_token_count = 0
    for chunk in chunk_batch:
        batch_token_count += count_tokens(chunk, tokenizer)
        progress_bar.update(1)
    return batch_token_count

# Function to process a decompressed dump file and count tokens with progress tracking
def process_dump(file_path, lang_code):
    tokenizer = get_tokenizer(lang_code)
    num_threads = multiprocessing.cpu_count()  # Use the number of CPU cores

    try:
        context = ET.iterparse(file_path, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        page_count = sum(1 for event, elem in context if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page')

        with tqdm(total=page_count, desc=f"Processing {lang_code}") as pbar:
            context = ET.iterparse(file_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            text_chunks = []
            for event, elem in context:
                if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.11/}page':
                    revision = elem.find('{http://www.mediawiki.org/xml/export-0.11/}revision')
                    if revision is not None:
                        text_elem = revision.find('{http://www.mediawiki.org/xml/export-0.11/}text')
                        if text_elem is not None and text_elem.text is not None:
                            # Split text into smaller chunks to avoid tokenizer issues
                            chunks = [text_elem.text[i:i+500] for i in range(0, len(text_elem.text), 500)]
                            text_chunks.extend(chunks)
                    pbar.update(1)
                    root.clear()  # Clear the root to free memory

            chunk_size = len(text_chunks) // num_threads
            if len(text_chunks) % num_threads != 0:
                chunk_size += 1

            chunk_batches = [text_chunks[i:i+chunk_size] for i in range(0, len(text_chunks), chunk_size)]
            print(f"Tokenizing {len(text_chunks)} chunks in parallel using {num_threads} threads...")

            # Tokenize text chunks in parallel with individual progress bars
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {}
                with tqdm(total=len(chunk_batches), desc="Threads Progress") as thread_pbar:
                    for batch in chunk_batches:
                        batch_pbar = tqdm(total=len(batch), desc="Batch Progress", leave=False)
                        future = executor.submit(process_chunk_batch, batch, tokenizer, batch_pbar)
                        futures[future] = batch_pbar

                    for future in as_completed(futures):
                        try:
                            tokens = future.result()
                            token_counts[lang_code] += tokens
                            futures[future].close()
                            thread_pbar.update(1)
                            print(f"Batch processed. Current token count for {lang_code}: {token_counts[lang_code]}")
                        except Exception as e:
                            print(f"Error processing batch: {e}")

        print(f"Processed {file_path}: {token_counts[lang_code]} tokens")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Log the problematic file for further investigation
        with open(f"{file_path}.error.log", 'w') as log_file:
            log_file.write(f"Error processing {file_path}: {e}\n")

# Function to handle the complete process for each language
def handle_language(lang_code, lang_name):
    print(f"Processing {lang_name} ({lang_code})")
    dump_path = download_dump(lang_code)
    if dump_path:
        decompressed_file_path = decompress_bz2(dump_path)
        process_dump(decompressed_file_path, lang_code)
        # Print the token count for the language
        print(f"{lang_name} ({lang_code}): {token_counts[lang_code]} tokens")
        # Save token counts to a file
        with open('token_counts.txt', 'a') as f:
            f.write(f"{language_map[lang_code]} ({lang_code}): {token_counts[lang_code]}\n")
        # Delete the dump files after processing
        os.remove(dump_path)
        os.remove(decompressed_file_path)
        print(f"Deleted {dump_path} and {decompressed_file_path}")

if __name__ == "__main__":
    # Read existing token counts to skip processed languages
    completed_languages = {}
    if os.path.exists('token_counts.txt'):
        with open('token_counts.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    lang_code = parts[0].split('(')[1].split(')')[0].strip()
                    token_count = int(parts[1].strip().split()[0])
                    completed_languages[lang_code] = token_count

    # Process one language at a time
    for lang_code, lang_name in tqdm(language_map.items(), desc="Overall Progress"):
        if lang_code in completed_languages and completed_languages[lang_code] > 0:
            print(f"Skipping {lang_name} ({lang_code}) as it is already processed with token count {completed_languages[lang_code]}.")
            continue
        handle_language(lang_code, lang_name)

    print("All languages processed. Token counts saved to token_counts.txt")
