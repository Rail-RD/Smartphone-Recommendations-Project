import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# Обработка данных

def extract_num(text):
    if pd.isna(text): return 0
    # Убираем лишние символы, которые мешают конвертации
    clean_text = str(text).replace(',', '').replace('$', '').strip()
    res = ""
    for c in clean_text:
        if c.isdigit() or (c == '.' and '.' not in res):
            res += c
        elif res:
            break
    return float(res) if res else 0



# Извлекает данные из названия модели, если в таблице пропуски
def extract_rom(model_name):
    if pd.isna(model_name): return 128.0
    words = str(model_name).upper().split()
    for w in words:
        if "GB" in w:  # Обработка гигабайт
            n = w.replace("GB", "")
            if n.replace('.', '').isdigit(): return float(n)
        if "TB" in w:  # Обработка терабайт
            n = w.replace("TB", "")
            if n.replace('.', '').isdigit(): return float(n) * 1024
    return 128.0



df = pd.read_csv('Mobiles_Dataset.csv', encoding='latin-1')

# Применение функций для создания числовых векторов
df['RAM_gb'] = df['RAM'].apply(extract_num)
df['ROM_gb'] = df['Model Name'].apply(extract_rom)
df['Price_usd'] = df['Launched Price (USA)'].apply(extract_num)
df['Year'] = df['Launched Year'].apply(extract_num)
df['Battery'] = df['Battery Capacity'].apply(extract_num)

# Модели 2010 года и ранее не актуальны для рекомендаций смартфона в 2025
df = df[(df['Price_usd'] > 0) & (df['Year'] > 2010)].reset_index(drop=True)


# Обучение модели

features = ['RAM_gb', 'ROM_gb', 'Price_usd', 'Year', 'Battery']

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Инициализация алгоритма K-Nearest Neighbors

model = NearestNeighbors(n_neighbors=10, metric='euclidean')
model.fit(X_scaled)


with open('processed_data.pkl', 'wb') as f: pickle.dump(df, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('phone_model.pkl', 'wb') as f: pickle.dump(model, f)






from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загрузка сохраненных состояний
df = pickle.load(open('processed_data.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('phone_model.pkl', 'rb'))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
        request: Request,
        min_price: float = Form(...),
        max_price: float = Form(...),
        ram: float = Form(...),
        rom: float = Form(...),
        year: int = Form(...)
):
    # Фильтр (Rule-based filtering)

    mask = (df['Price_usd'] >= min_price) & (df['Price_usd'] <= max_price) & \
           (df['RAM_gb'] >= ram) & (df['ROM_gb'] >= rom)
    if year > 0: mask &= (df['Year'] >= year)

    filtered_df = df[mask].copy()

    # Проверка на пустой результат (валидация пользовательского ввода)
    if filtered_df.empty:
        return templates.TemplateResponse("index.html", {
            "request": request, "error": "Моделей с такими параметрами не найдено.", "results": None
        })

    # Использование обученной модели (Machine Learning)
    # Создаем точку "идеального телефона" на основе желаний пользователя.

    user_features = np.array([[ram, rom, (min_price + max_price) / 2, (year if year > 0 else 2023), 5000]])

    user_scaled = scaler.transform(user_features)

    # Поиск 10 ближайших соседей
    distances, indices = model.kneighbors(user_scaled)


    # Сопоставляем "умные" рекомендации модели с  фильтром пользователя.
    recommended_indices = [idx for idx in indices[0] if idx in filtered_df.index]

    # Если модель нашла совпадения в рамках фильтра — показываем их.
    # Если нет — просто выводим лучшие по году выпуска из отфильтрованных.
    if recommended_indices:
        final_df = df.iloc[recommended_indices]
    else:
        final_df = filtered_df.sort_values(by='Year', ascending=False).head(10)



    # Конвертация таблицы в формат JSON/Dictionary для отображения в HTML
    results = final_df.to_dict(orient='records')
    return templates.TemplateResponse("index.html", {
        "request": request, "results": results, "search_params": f"${min_price}-${max_price}"
    })




















































