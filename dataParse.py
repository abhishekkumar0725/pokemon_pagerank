import pandas as pd

def main():
    types = pd.read_csv('type.csv')
    df =  pd.DataFrame(columns = ['Attacking', 'Defending', 'Effectiveness'])
    for index, row in types.iterrows():
        for type in types.columns:
            if type == 'Attacking':
                continue
            df = df.append({ 'Attacking': row['Attacking'],
                        'Defending': type,
                        'Effectiveness': row[type]}, ignore_index=True)
        
    df.to_csv('effectiveness.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    main()