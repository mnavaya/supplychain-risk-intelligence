"""Backward-compatible entrypoint. Prefer: python main.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import SupplyChainAgent, run_pipeline  # noqa: E402
import time
import datetime

if __name__ == "__main__":
    start_time = time.time()
    print(
        f"🚀 Supply Chain Intelligence Pipeline Active - "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    result = run_pipeline("week_2026-04-13_demo.csv")

    if result is not None:
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"🎉 ENGINE EXECUTION COMPLETE in {total_time:.1f} seconds")
        print("=" * 60)

        print("\n🤖 Inventory Copilot Online! (Type 'exit' to log out)")
        agent = SupplyChainAgent(result["full_summary"])
        while True:
            try:
                user_query = input("\n💬 Copilot Command: ")
                if user_query.lower() in ["exit", "quit"]:
                    print("Logging out safely. Have a great shift! 📦")
                    break
                if user_query.strip():
                    print(agent.query(user_query))
            except (KeyboardInterrupt, EOFError):
                break
