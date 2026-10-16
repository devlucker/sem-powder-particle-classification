from __future__ import annotations

import json

from powder_ml import MODEL_PATH, save_model


def main() -> None:
    trained = save_model()
    print(f"Saved trained material classifier to {MODEL_PATH}")
    print(json.dumps(trained["metrics"], indent=2))


if __name__ == "__main__":
    main()
