def statistic(texts):
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
    print(f"\nСамая длинная строка ({max_len} символов)\n Самый длинный текст {max_text}")  