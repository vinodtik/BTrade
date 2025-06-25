import pandas as pd
import re
from config import CSV_COLUMNS

def load_option_chain_csv(filepath):
    """
    Custom loader for option chain CSV with combined CALL/PUT rows.
    Skips the first two header rows and parses data from the third row onward.
    """
    # Read the file, skipping the first row (CALLS,,PUTS)
    raw = pd.read_csv(filepath, header=None, skiprows=1)
    
    # Get column names from the first row
    columns = raw.iloc[0]
    raw = raw[1:]  # Remove the header row
    records = []
    
    # Get the spot price from the underlying price row if available
    try:
        spot_price = None
        for idx, row in raw.iterrows():
            if 'Underlying Price' in str(row[0]):
                spot_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', str(row[0]))
                if spot_match:
                    spot_price = float(spot_match.group(1).replace(',', ''))
                break
    except:
        spot_price = None
    
    for idx, row in raw.iterrows():
        try:
            # Skip rows without valid strike price
            if pd.isna(row[11]) or str(row[11]).strip() == '' or 'Underlying' in str(row[0]):
                continue
                
            strike = float(str(row[11]).replace(',', ''))
            
            # CALL side
            call = {
                'Strike Price': strike,
                'Option Type': 'CALL',
                'LTP': float(str(row[5]).replace(',', '').replace('-', '0')) if pd.notna(row[5]) and str(row[5]).strip() != '' else 0.0,
                'Bid': float(str(row[8]).replace(',', '').replace('-', '0')) if pd.notna(row[8]) and str(row[8]).strip() != '' else 0.0,
                'Ask': float(str(row[9]).replace(',', '').replace('-', '0')) if pd.notna(row[9]) and str(row[9]).strip() != '' else 0.0,
                'OI': int(str(row[1]).replace(',', '').replace('-', '0')) if pd.notna(row[1]) and str(row[1]).strip() != '' else 0,
                'Change in OI': int(str(row[2]).replace(',', '').replace('-', '0')) if pd.notna(row[2]) and str(row[2]).strip() != '' else 0,
                'Volume': int(str(row[3]).replace(',', '').replace('-', '0')) if pd.notna(row[3]) and str(row[3]).strip() != '' else 0,
                'IV': float(str(row[4]).replace(',', '').replace('-', '0')) if pd.notna(row[4]) and str(row[4]).strip() != '' and str(row[4]).replace('.', '').replace('-', '').isdigit() else 0.0,
                'Spot Price': spot_price
            }
            records.append(call)
            
            # PUT side
            put = {
                'Strike Price': strike,
                'Option Type': 'PUT',
                'LTP': float(str(row[17]).replace(',', '').replace('-', '0')) if pd.notna(row[17]) and str(row[17]).strip() != '' else 0.0,
                'Bid': float(str(row[13]).replace(',', '').replace('-', '0')) if pd.notna(row[13]) and str(row[13]).strip() != '' else 0.0,
                'Ask': float(str(row[14]).replace(',', '').replace('-', '0')) if pd.notna(row[14]) and str(row[14]).strip() != '' else 0.0,
                'OI': int(str(row[21]).replace(',', '').replace('-', '0')) if pd.notna(row[21]) and str(row[21]).strip() != '' else 0,
                'Change in OI': int(str(row[20]).replace(',', '').replace('-', '0')) if pd.notna(row[20]) and str(row[20]).strip() != '' else 0,
                'Volume': int(str(row[19]).replace(',', '').replace('-', '0')) if pd.notna(row[19]) and str(row[19]).strip() != '' else 0,
                'IV': float(str(row[18]).replace(',', '').replace('-', '0')) if pd.notna(row[18]) and str(row[18]).strip() != '' and str(row[18]).replace('.', '').replace('-', '').isdigit() else 0.0,
                'Spot Price': spot_price
            }
            records.append(put)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
            
    df = pd.DataFrame(records)
    
    # If spot price wasn't found, use ATM strike as an approximation
    if not df.empty and (spot_price is None or pd.isna(spot_price)):
        spot_price = df['Strike Price'].median()
        
    if not df.empty:
        df['Spot Price'] = spot_price
        
    return df
