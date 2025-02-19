import aiohttp
import asyncio


async def test(data):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://127.0.0.1:8080/gradio_api/call/predict',
            json={
                "data": [
                    str(data)
                ]
            }
        ) as resp:
            print(await resp.json())
            print(f'attempt_{data}')


async def main():
    await asyncio.gather(*[test(i) for i in range(10)])


asyncio.run(main())