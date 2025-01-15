import json

if __name__ == '__main__':
    data = []
    with open('data/data.txt', 'r') as file:
        text_data = file.read()

    text_data = text_data.split('\n\n')
    for i in range(len(text_data)):
        data.append({
            'title': text_data[i].split('\n')[0],
            'content': text_data[i]
        })

    with open('data/data.json', 'w') as file:
        json.dump(data, file, indent=4)