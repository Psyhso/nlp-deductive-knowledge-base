"""Start the local application with `python -m factlab` or `factlab`."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Лаборатория фактов: NLP и логический вывод")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес интерфейса")
    parser.add_argument("--port", type=int, default=8000, help="Порт HTTP-сервера")
    parser.add_argument("--reload", action="store_true", help="Перезапуск при изменении кода")
    args = parser.parse_args()
    uvicorn.run("factlab.web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
