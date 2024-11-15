import os
from collections import Counter


def count_files_with_same_prefix(folder_path):
    """Összeszámolja azokat a fájlokat, amelyek azonos prefixszel rendelkeznek, a végükben eltérő (amp, mask, phase)."""
    if not os.path.exists(folder_path):
        print(f"A mappa nem található: {folder_path}")
        return None

    # Fájlok beolvasása a mappából
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  os.path.isfile(os.path.join(folder_path, f))]

    # Prefixek kinyerése (a végződés nélkül)
    prefixes = [os.path.basename(f).rsplit('_', 1)[0] for f in file_paths]

    # Prefixek számlálása
    prefix_counts = Counter(prefixes)

    return prefix_counts


def main():
    """Main program, amely összeszámolja az azonos prefixű fájlokat."""
    folder_path = "./augmentation"

    prefix_counts = count_files_with_same_prefix(folder_path)

    if prefix_counts:
        print("\nAz azonos prefixű fájlok száma (amp, mask, phase figyelembe véve):")
        for prefix, count in prefix_counts.items():
            if count != 3 :
                print(f"Prefix: {prefix}, Count: {count}")
    else:
        print("Nincsenek fájlok a megadott mappában vagy a mappa nem létezik.")

    print("vegeeee")


# A program futtatása
if __name__ == "__main__":
    main()
