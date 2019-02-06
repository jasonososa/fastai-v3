from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1w8F8rb8Ty4R_BkZEhiHs7BpqRyhXm874'
export_file_name = 'export.pkl'

# Actual classes predicted by the model
predicted_classes = ['Dolls_less_than 250','Dolls_between_250_and_1000','Dolls_more_than_1000']
# Desire output classs
classes = ['Value of the doll is less than $250', 'Value of the doll is between $250 and $1,0000',
           'Value of the doll is more than $1,000']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    learn = load_learner(path, export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    output_class = []
    if prediction=='Dolls_less_than 250' :
        output_class.append('Value of the doll is less than $250')
    elif prediction=='Dolls_between_250_and_1000' :
        output_class.append('Value of the doll is between $250 and $1,000')
    elif prediction=='Dolls_more_than_1000' :
        output_class.append('Value of the doll is more than $1,000')
    return JSONResponse({'result': str(output_class)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
