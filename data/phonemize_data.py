import subprocess
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
import warnings
from tqdm import tqdm 

warnings.filterwarnings("ignore", message="Trying to detect language from a single word.")

languages = [Language.ENGLISH, Language.INDONESIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

@lru_cache(maxsize=100_000)
def detect_lang(word: str) -> str:
    result = detector.detect_language_of(word)
    if result is None:
        return "id"
    return "en" if result == Language.ENGLISH else "id"

@lru_cache(maxsize=100_000)
def phonemize_word(word: str, ipa: bool, keep_stress: bool, sep: str) -> str:
    lang = detect_lang(word)
    lang_map = {"id": "id", "en": "en-us"}
    voice = lang_map.get(lang, "id")
    cmd = ["espeak-ng", "-v", voice, "-q", f"--sep={sep}", word]
    if ipa:
        cmd.insert(3, "--ipa")
    else:
        cmd.insert(3, "-x")
    try:
        result = subprocess.run(cmd, capture_output=True)
        phonemes = result.stdout.decode("utf-8", errors="ignore").strip()
        phonemes = phonemes.replace("\ufeff", "")
        if not keep_stress:
            phonemes = re.sub(r"[ˈˌ]", "", phonemes)
        return phonemes
    except (subprocess.TimeoutExpired, Exception):
        return word

class EnIndPhonemizer:
    def __init__(self, ipa=True, keep_stress=False, sep="", max_workers=None):
        self.ipa = ipa
        self.keep_stress = keep_stress
        self.sep = sep
        self.max_workers = max_workers or 4

    def phonemize(self, text: str) -> str:
        if not text:
            return ""
        words = text.strip().split()
        phonemized_words = [
            phonemize_word(w, self.ipa, self.keep_stress, self.sep) for w in words
        ]
        return " ".join(phonemized_words)

    def process_in_parallel(self, texts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(
                tqdm(
                    executor.map(self.phonemize, texts),
                    total=len(texts),
                    desc="Phonemizing Sentences"
                )
            )
    
def main():
    source_csv = Path("data/data/wavs/metadata.csv")
    output_directory = Path("data/data")

    try:
        df = pd.read_csv(source_csv)
    except FileNotFoundError:
        print(f"Error: Source file not found at '{source_csv}'")
        return

    df['wav_path'] = df['filename'].apply(lambda x: f"data/data/wavs/{x}")
    df['grapheme'] = df['text'].str.strip()
    df['speaker_id'] = df['encoded_speaker']

    phonemizer = EnIndPhonemizer(ipa=True, keep_stress=True)
    
    df['phoneme'] = phonemizer.process_in_parallel(df['grapheme'].tolist())
    
    df['grapheme_line'] = df['wav_path'] + '|' + df['grapheme'] + '|' + df['speaker_id'].astype(str)
    df['phoneme_line'] = df['wav_path'] + '|' + df['phoneme'] + '|' + df['speaker_id'].astype(str)
    
    output_directory.mkdir(exist_ok=True)
    grapheme_file = output_directory / "metadata_grapheme.txt"
    phoneme_file = output_directory / "metadata_phoneme.txt"
    
    df['grapheme_line'].to_csv(grapheme_file, index=False, header=False, encoding="utf-8")
    df['phoneme_line'].to_csv(phoneme_file, index=False, header=False, encoding="utf-8")
    
    print(f"\nCreated files:")
    print(f"- {grapheme_file}")
    print(f"- {phoneme_file}")

if __name__ == "__main__":
    main()