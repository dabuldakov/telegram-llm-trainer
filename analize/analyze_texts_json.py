import json

def analyze_texts_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from collections import defaultdict

    stats = defaultdict(lambda: {"count": 0, "context_len": 0, "target_len": 0, "total_len": 0})

    for item in data:
        author = item.get("author") or "Unknown"
        context = item.get("context") or ""
        target = item.get("target") or ""
        context_len = len(context)
        target_len = len(target)
        total_len = context_len + target_len

        stats[author]["count"] += 1
        stats[author]["context_len"] += context_len
        stats[author]["target_len"] += target_len
        stats[author]["total_len"] += total_len

    print(f"{'Автор':30} {'Кол-во':>7} {'Ср. контекст':>12} {'Ср. ответ':>10} {'Ср. блок':>10}")
    print("-" * 75)
    for author, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
        count = s["count"]
        avg_context = s["context_len"] // count if count else 0
        avg_target = s["target_len"] // count if count else 0
        avg_total = s["total_len"] // count if count else 0
        print(f"{author:30} {count:7} {avg_context:12} {avg_target:10} {avg_total:10}")

    all_blocks = []
    for item in data:
        # Можно анализировать context, target или их объединение
        block = (item.get("context") or "") + (item.get("target") or "")
        all_blocks.append(block)
    statistic(all_blocks)    


def statistic(texts):
    # Расширенные диапазоны по длине сообщений
    bins = {
        "0-100": 0,
        "100-250": 0,
        "250-500": 0,
        "500-1000": 0,
        "1000-1500": 0,
        "1500-2000": 0,
        "2000-2500": 0,
        "2500-3000": 0,
        "3000-3500": 0,
        "3500-4000": 0,
        "4000-4500": 0,
        "4500-5000": 0,
        "5000+": 0
    }
    max_len = 0
    max_text = ""
    for text in texts:
        l = len(text)
        if l > max_len:
            max_len = l
            max_text = text
        if l <= 100:
            bins["0-100"] += 1
        elif l <= 250:
            bins["100-250"] += 1
        elif l <= 500:
            bins["250-500"] += 1
        elif l <= 1000:
            bins["500-1000"] += 1
        elif l <= 1500:
            bins["1000-1500"] += 1
        elif l <= 2000:
            bins["1500-2000"] += 1
        elif l <= 2500:
            bins["2000-2500"] += 1
        elif l <= 3000:
            bins["2500-3000"] += 1
        elif l <= 3500:
            bins["3000-3500"] += 1
        elif l <= 4000:
            bins["3500-4000"] += 1
        elif l <= 4500:
            bins["4000-4500"] += 1
        elif l <= 5000:
            bins["4500-5000"] += 1
        else:
            bins["5000+"] += 1

    print("Статистика по длине сообщений:")
    for k, v in bins.items():
        print(f"{k}: {v}")
    print(f"\nСамая длинная строка ({max_len} символов)")    

# Пример использования:
analyze_texts_json("data/chat_history_prepared.json")