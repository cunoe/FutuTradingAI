import futu as ft
import pandas as pd
import os
from openai import OpenAI

# --- AI Configuration ---
TUZI_API_KEY = os.environ.get("TUZI_API_KEY")
TUZI_BASE_URL = os.environ.get("TUZI_BASE_URL")

try:
    if TUZI_API_KEY and TUZI_BASE_URL:
        ai_client = OpenAI(
            api_key=TUZI_API_KEY,
            base_url=TUZI_BASE_URL,
        )
        AI_ENABLED = True
    else:
        ai_client = None
        AI_ENABLED = False
except Exception as e:
    ai_client = None
    AI_ENABLED = False


# --- Connection Configuration ---
FUTU_HOST = '127.0.0.1'
FUTU_PORT = 11111
TRADING_ENVIRONMENT = ft.TrdEnv.REAL # Use ft.TrdEnv.SIMULATE for paper trading

def print_dataframe_as_table(df: pd.DataFrame | pd.Series):
    """Prints a pandas DataFrame or Series in a well-aligned table format."""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        return
        
    df_str = df.astype(str)
    
    # Calculate column widths
    column_widths = {}
    for col in df_str.columns:
        # Get the max width between column header and column values
        max_len = max(len(col), df_str[col].str.len().max() if not df_str[col].empty else 0)
        column_widths[col] = max_len

    # Determine alignment for each column (right for numeric, left for others)
    alignments = {
        col: '>' if pd.api.types.is_numeric_dtype(df[col].dtype) else '<'
        for col in df.columns
    }

    # Print header
    header_line = " | ".join(f"{name.center(width)}" for name, width in zip(df.columns, column_widths.values()))
    print(f"  {header_line}")

    # Print separator
    separator_line = "-+-".join("-" * width for width in column_widths.values())
    print(f"  {separator_line}")

    # Print rows
    for _, row in df_str.iterrows():
        row_values = []
        for col, value in row.items():
            align_char = alignments.get(col, '<')
            row_values.append(f"{value:{align_char}{column_widths[col]}}")
        
        line = " | ".join(row_values)
        print(f"  {line}")

def ask_ai(client: OpenAI, acc_info_df: pd.DataFrame, pos_df: pd.DataFrame):
    """Asks a question to the AI with account context."""
    if not AI_ENABLED:
        print("AI client is not configured. Please set the TUZI_API_KEY and TUZI_BASE_URL environment variables.")
        return

    print("\nAsk a question about your account to the AI assistant.")
    print("Type 'quit' or 'q' to return to the previous menu.")
    user_question = input("Your question: ")

    if user_question.lower() in ['quit', 'q']:
        return

    acc_summary_md = acc_info_df.to_markdown(index=False)
    positions_md = pos_df.to_markdown(index=False) if not pos_df.empty else "No positions."

    prompt_context = f"""
You are a professional financial assistant from TUZI AI. Based on the following real-time account information, please answer the user's question.

### Account Summary
{acc_summary_md}

### Current Positions
{positions_md}
"""

    try:
        print("\n🤖 Thinking...")
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt_context},
                {"role": "user", "content": user_question},
            ],
            stream=True,
        )
        print("✅ AI Response:")
        for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")
        print("\n")
    except Exception as e:
        print(f"\nAn error occurred while communicating with the AI service: {e}")


def main():
    """
    Connects to FutuOpenD, fetches and prints account and position information.
    """
    print("Connecting to FutuOpenD via WebSocket...")
    
    # For older futu-api versions, passing connection parameters directly is the way.
    trade_context = ft.OpenSecTradeContext(
        host=FUTU_HOST, 
        port=FUTU_PORT, 
        security_firm=ft.SecurityFirm.FUTUSECURITIES
    )
    
    try:
        # 1. Get account list
        print("\nFetching account list...")
        ret_code, acc_list_data = trade_context.get_acc_list()
        if ret_code != ft.RET_OK:
            print(f"Error fetching account list: {acc_list_data}")
            return
        
        if not isinstance(acc_list_data, pd.DataFrame) or acc_list_data.empty:
            print("No accounts found.")
            return

        print(f"Found {len(acc_list_data)} accounts.\n")

        # 2. Iterate through each account to get details and positions
        while True:
            print("Please select an account to view details:")
            for idx, (_, acc_row) in enumerate(acc_list_data.iterrows()):
                acc_id_display = int(acc_row['acc_id'])
                trd_env_display = str(acc_row['trd_env'])
                print(f"  {idx + 1}: Account ID {acc_id_display} ({trd_env_display})")
            
            print("  0: Exit")

            try:
                choice_str = input("Enter your choice: ")
                choice_idx = int(choice_str)

                if choice_idx == 0:
                    break
                
                if choice_idx < 1 or choice_idx > len(acc_list_data):
                    print("Invalid choice, please try again.\n")
                    continue

                selected_account = acc_list_data.iloc[choice_idx - 1]
                acc_id = int(selected_account['acc_id'])
                trd_env = str(selected_account['trd_env'])
                
                print(f"\n--- Processing Account ID: {acc_id} (Env: {trd_env}) ---")

                # Get account information for this specific account
                ret_acc_info, data_acc_info = trade_context.accinfo_query(trd_env=trd_env, acc_id=acc_id)

                if ret_acc_info != ft.RET_OK:
                    print(f"  Error fetching account info for {acc_id}: {data_acc_info}")
                elif isinstance(data_acc_info, pd.DataFrame) and not data_acc_info.empty:
                    print(f"  Account Summary:")
                    
                    display_columns = ['total_assets', 'cash', 'market_val', 'unrealized_pl', 'realized_pl']
                    
                    output_df = data_acc_info.copy()
                    for col in display_columns:
                        if col not in output_df.columns:
                            output_df[col] = 'N/A'
                    
                    print_dataframe_as_table(output_df[display_columns])
                else:
                    print(f"  No account information found for account {acc_id}.")

                # Get positions for this specific account
                ret_pos, data_pos = trade_context.position_list_query(trd_env=trd_env, acc_id=acc_id)

                if ret_pos != ft.RET_OK:
                    print(f"  Error fetching positions for account {acc_id}: {data_pos}")
                elif isinstance(data_pos, pd.DataFrame) and not data_pos.empty:
                    print(f"\n  Positions:")
                    pos_display_columns = [
                        'code', 'stock_name', 'qty', 'can_sell_qty', 'price', 
                        'cost_price', 'market_val', 'pl_ratio', 'pl_val'
                    ]
                    for col in pos_display_columns:
                        if col not in data_pos.columns:
                            data_pos[col] = 'N/A'

                    print_dataframe_as_table(data_pos[pos_display_columns])
                else:
                    print(f"  No positions found for account {acc_id}.")
                print("----------------------------------------------------------\n")

                if not AI_ENABLED:
                    print("\n[AI Assistant is disabled. Set TUZI_API_KEY and TUZI_BASE_URL to enable it.]")


                # --- Account-specific action loop ---
                while True:
                    print(f"\n--- Account Menu for {acc_id} ---")
                    print("1: View Account Summary & Positions")
                    if AI_ENABLED:
                        print("2: Ask AI Assistant")
                    print("0: Go back to account list")

                    action_choice = input("Select an action: ")

                    if action_choice == '1':
                        # Get account information for this specific account
                        ret_acc_info, data_acc_info = trade_context.accinfo_query(trd_env=trd_env, acc_id=acc_id)

                        if ret_acc_info != ft.RET_OK:
                            print(f"  Error fetching account info for {acc_id}: {data_acc_info}")
                        elif isinstance(data_acc_info, pd.DataFrame) and not data_acc_info.empty:
                            print(f"\n  Account Summary:")
                            
                            display_columns = ['total_assets', 'cash', 'market_val', 'unrealized_pl', 'realized_pl']
                            
                            output_df = data_acc_info.copy()
                            for col in display_columns:
                                if col not in output_df.columns:
                                    output_df[col] = 'N/A'
                            
                            print_dataframe_as_table(output_df[display_columns])
                        else:
                            print(f"  No account information found for account {acc_id}.")

                        # Get positions for this specific account
                        ret_pos, data_pos = trade_context.position_list_query(trd_env=trd_env, acc_id=acc_id)

                        if ret_pos != ft.RET_OK:
                            print(f"  Error fetching positions for account {acc_id}: {data_pos}")
                        elif isinstance(data_pos, pd.DataFrame) and not data_pos.empty:
                            print(f"\n  Positions:")
                            pos_display_columns = [
                                'code', 'stock_name', 'qty', 'can_sell_qty', 'price', 
                                'cost_price', 'market_val', 'pl_ratio', 'pl_val'
                            ]
                            for col in pos_display_columns:
                                if col not in data_pos.columns:
                                    data_pos[col] = 'N/A'

                            print_dataframe_as_table(data_pos[pos_display_columns])
                        else:
                            print(f"  No positions found for account {acc_id}.")
                        print("----------------------------------------------------------\n")
                    
                    elif action_choice == '2' and AI_ENABLED:
                        if not ai_client:
                            print("AI client is not properly configured. Please check your environment variables.")
                            continue
                        # Ensure we have fresh data before asking AI
                        ret_acc_info, data_acc_info = trade_context.accinfo_query(trd_env=trd_env, acc_id=acc_id)
                        ret_pos, data_pos = trade_context.position_list_query(trd_env=trd_env, acc_id=acc_id)

                        if ret_acc_info == ft.RET_OK and ret_pos == ft.RET_OK and isinstance(data_acc_info, pd.DataFrame) and isinstance(data_pos, pd.DataFrame):
                            ask_ai(ai_client, data_acc_info, data_pos)
                        else:
                            print("Could not fetch latest account data to provide to AI.")
                            if ret_acc_info != ft.RET_OK:
                                print(f"  Error fetching account info: {data_acc_info}")
                            if ret_pos != ft.RET_OK:
                                print(f"  Error fetching positions: {data_pos}")

                    elif action_choice == '0':
                        break # Exit account menu
                    else:
                        print("Invalid choice, please try again.")

            except ValueError:
                print("Invalid input. Please enter a number.\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Closing connection.")
        trade_context.close()

if __name__ == "__main__":
    main()
