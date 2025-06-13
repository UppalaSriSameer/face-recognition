import os
import shutil
import argparse

def people(root):
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def count_images(path):
    exts = (".jpg", ".jpeg", ".png")
    try:
        return sum(1 for f in os.listdir(path) if f.lower().endswith(exts))
    except FileNotFoundError:
        return 0

def show_list(root):
    ps = people(root)
    if not ps:
        print("[INFO] No dataset found.")
        return
    print(f"[INFO] Dataset at: {os.path.abspath(root)}")
    for p in ps:
        n = count_images(os.path.join(root, p))
        print(f" - {p}: {n} images")

def confirm(msg, yes):
    if yes:
        return True
    r = input(f"{msg} [y/N]: ").strip().lower()
    return r in ("y", "yes")

def delete_person(root, name, yes):
    ps = people(root)
    if name not in ps:
        print(f"[ERROR] '{name}' not found.")
        if ps:
            print("[INFO] Available: " + ", ".join(ps))
        else:
            print("[INFO] No people in dataset.")
        return
    target = os.path.join(root, name)
    if confirm(f"Delete all samples for '{name}' at {target}?", yes):
        shutil.rmtree(target)
        print(f"[DONE] Deleted samples for '{name}'.")
        print("[NOTE] Re-train the model:  python create_classifier.py")
    else:
        print("[CANCELLED] No changes made.")

def delete_all(root, yes):
    ps = people(root)
    if not ps:
        print("[INFO] No people to delete.")
        return
    if confirm(f"Delete ALL people in dataset at {os.path.abspath(root)}?", yes):
        for p in ps:
            shutil.rmtree(os.path.join(root, p), ignore_errors=True)
        print("[DONE] Deleted all people.")
        print("[NOTE] Collect fresh samples and re-train.")
    else:
        print("[CANCELLED] No changes made.")

def main():
    parser = argparse.ArgumentParser(description="List or delete samples in the dataset.")
    parser.add_argument("--dataset_dir", default=os.path.join("..", "dataset"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")

    p_del = sub.add_parser("delete")
    p_del.add_argument("--name", required=True)
    p_del.add_argument("--yes", action="store_true")

    p_all = sub.add_parser("delete-all")
    p_all.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    root = os.path.abspath(args.dataset_dir)

    if args.cmd == "list":
        show_list(root)
    elif args.cmd == "delete":
        delete_person(root, args.name, args.yes)
    elif args.cmd == "delete-all":
        delete_all(root, args.yes)

if __name__ == "__main__":
    main()