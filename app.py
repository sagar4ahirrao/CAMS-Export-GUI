import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import base64
from io import BytesIO
import warnings
import casparser
import logging
import time
import os
import traceback
import requests
import json
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cams_analyzer.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database operations for portfolio data"""
    
    def __init__(self, db_path: str = "portfolio_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create portfolios table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Create holdings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS holdings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER,
                        fund_name TEXT NOT NULL,
                        isin TEXT,
                        amfi TEXT,
                        rta TEXT,
                        rta_code TEXT,
                        advisor TEXT,
                        type TEXT,
                        units REAL,
                        market_value REAL,
                        cost_value REAL,
                        nav REAL,
                        folio TEXT,
                        amc TEXT,
                        pan TEXT,
                        kyc TEXT,
                        pankyc TEXT,
                        current_nav REAL,
                        nav_change_percent REAL,
                        current_value REAL,
                        absolute_gain REAL,
                        return_percentage REAL,
                        todays_gain_value REAL,
                        todays_gain_percent REAL,
                        portfolio_weight REAL,
                        avg_purchase_price REAL,
                        fund_house TEXT,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                ''')
                
                # Create transactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER,
                        date TIMESTAMP,
                        description TEXT,
                        amount REAL,
                        units REAL,
                        nav REAL,
                        balance REAL,
                        type TEXT,
                        scheme TEXT,
                        folio TEXT,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                ''')
                
                # Create portfolio_summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id INTEGER,
                        total_market_value REAL,
                        total_cost REAL,
                        statement_period_from TEXT,
                        statement_period_to TEXT,
                        processing_time REAL,
                        FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def save_portfolio(self, filename: str, file_content: bytes, parser: 'CAMSParser') -> int:
        """Save portfolio data to database"""
        try:
            file_hash = self.calculate_file_hash(file_content)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if portfolio already exists
                cursor.execute('SELECT id FROM portfolios WHERE filename = ?', (filename,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing portfolio
                    portfolio_id = existing[0]
                    cursor.execute('''
                        UPDATE portfolios 
                        SET file_hash = ?, last_updated = CURRENT_TIMESTAMP, status = 'active'
                        WHERE id = ?
                    ''', (file_hash, portfolio_id))
                    
                    # Delete existing holdings and transactions
                    cursor.execute('DELETE FROM holdings WHERE portfolio_id = ?', (portfolio_id,))
                    cursor.execute('DELETE FROM transactions WHERE portfolio_id = ?', (portfolio_id,))
                    cursor.execute('DELETE FROM portfolio_summary WHERE portfolio_id = ?', (portfolio_id,))
                    
                    logger.info(f"Updated existing portfolio: {filename}")
                else:
                    # Insert new portfolio
                    cursor.execute('''
                        INSERT INTO portfolios (filename, file_hash)
                        VALUES (?, ?)
                    ''', (filename, file_hash))
                    portfolio_id = cursor.lastrowid
                    logger.info(f"Created new portfolio: {filename}")
                
                # Calculate performance metrics before saving
                holdings_df = pd.DataFrame(parser.holdings)
                if not holdings_df.empty:
                    holdings_df = calculate_performance_metrics(holdings_df)
                    # Update parser holdings with calculated values
                    parser.holdings = holdings_df.to_dict('records')
                
                # Save holdings
                for holding in parser.holdings:
                    # Ensure current_value is calculated
                    current_value = holding.get('current_value', 0)
                    if current_value == 0 and holding.get('market_value', 0) > 0:
                        current_value = holding.get('market_value', 0)
                    
                    cursor.execute('''
                        INSERT INTO holdings (
                            portfolio_id, fund_name, isin, amfi, rta, rta_code, advisor, type,
                            units, market_value, cost_value, nav, folio, amc, pan, kyc, pankyc,
                            current_nav, nav_change_percent, current_value, absolute_gain,
                            return_percentage, todays_gain_value, todays_gain_percent,
                            portfolio_weight, avg_purchase_price, fund_house
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        portfolio_id, holding.get('fund_name', ''), holding.get('isin', ''),
                        holding.get('amfi', ''), holding.get('rta', ''), holding.get('rta_code', ''),
                        holding.get('advisor', ''), holding.get('type', ''), holding.get('units', 0),
                        holding.get('market_value', 0), holding.get('cost_value', 0),
                        holding.get('nav', 0), holding.get('folio', ''), holding.get('amc', ''),
                        holding.get('pan', ''), holding.get('kyc', ''), holding.get('pankyc', ''),
                        holding.get('current_nav', 0), holding.get('nav_change_percent', 0),
                        current_value, holding.get('absolute_gain', 0),
                        holding.get('return_percentage', 0), holding.get('todays_gain_value', 0),
                        holding.get('todays_gain_percent', 0), holding.get('portfolio_weight', 0),
                        holding.get('avg_purchase_price', 0), holding.get('fund_house', '')
                    ))
                
                # Save transactions
                for transaction in parser.transactions:
                    cursor.execute('''
                        INSERT INTO transactions (
                            portfolio_id, date, description, amount, units, nav, balance, type, scheme, folio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        portfolio_id, transaction.get('date'), transaction.get('description', ''),
                        transaction.get('amount', 0), transaction.get('units', 0),
                        transaction.get('nav', 0), transaction.get('balance', 0),
                        transaction.get('type', ''), transaction.get('scheme', ''),
                        transaction.get('folio', '')
                    ))
                
                # Save portfolio summary
                cursor.execute('''
                    INSERT INTO portfolio_summary (
                        portfolio_id, total_market_value, total_cost, statement_period_from,
                        statement_period_to, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    portfolio_id,
                    parser.portfolio_summary.get('total_market_value', 0),
                    parser.portfolio_summary.get('total_cost', 0),
                    parser.portfolio_summary.get('statement_period', {}).get('from', ''),
                    parser.portfolio_summary.get('statement_period', {}).get('to', ''),
                    parser.processing_time
                ))
                
                conn.commit()
                logger.info(f"Successfully saved portfolio {filename} with {len(parser.holdings)} holdings and {len(parser.transactions)} transactions")
                return portfolio_id
                
        except Exception as e:
            logger.error(f"Error saving portfolio {filename}: {str(e)}")
            raise
    
    def load_all_portfolios(self) -> List[Dict]:
        """Load all active portfolios from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all active portfolios
                cursor.execute('''
                    SELECT id, filename, upload_date, last_updated, status
                    FROM portfolios 
                    WHERE status = 'active'
                    ORDER BY last_updated DESC
                ''')
                portfolios = cursor.fetchall()
                
                result = []
                for portfolio in portfolios:
                    portfolio_id, filename, upload_date, last_updated, status = portfolio
                    
                    # Load holdings
                    cursor.execute('''
                        SELECT * FROM holdings WHERE portfolio_id = ?
                    ''', (portfolio_id,))
                    holdings_data = cursor.fetchall()
                    holdings_columns = [description[0] for description in cursor.description]
                    
                    # Load transactions
                    cursor.execute('''
                        SELECT * FROM transactions WHERE portfolio_id = ?
                    ''', (portfolio_id,))
                    transactions_data = cursor.fetchall()
                    transactions_columns = [description[0] for description in cursor.description]
                    
                    # Load portfolio summary
                    cursor.execute('''
                        SELECT * FROM portfolio_summary WHERE portfolio_id = ?
                    ''', (portfolio_id,))
                    summary_data = cursor.fetchone()
                    summary_columns = [description[0] for description in cursor.description]
                    
                    # Convert to dictionaries
                    holdings = []
                    for holding in holdings_data:
                        holding_dict = dict(zip(holdings_columns, holding))
                        # Remove id and portfolio_id from holding dict
                        holding_dict.pop('id', None)
                        holding_dict.pop('portfolio_id', None)
                        holdings.append(holding_dict)
                    
                    transactions = []
                    for transaction in transactions_data:
                        trans_dict = dict(zip(transactions_columns, transaction))
                        # Remove id and portfolio_id from transaction dict
                        trans_dict.pop('id', None)
                        trans_dict.pop('portfolio_id', None)
                        transactions.append(trans_dict)
                    
                    summary = {}
                    if summary_data:
                        summary = dict(zip(summary_columns, summary_data))
                        summary.pop('id', None)
                        summary.pop('portfolio_id', None)
                    
                    result.append({
                        'id': portfolio_id,
                        'filename': filename,
                        'upload_date': upload_date,
                        'last_updated': last_updated,
                        'status': status,
                        'holdings': holdings,
                        'transactions': transactions,
                        'portfolio_summary': summary
                    })
                
                logger.info(f"Loaded {len(result)} portfolios from database")
                return result
                
        except Exception as e:
            logger.error(f"Error loading portfolios: {str(e)}")
            return []
    
    def delete_portfolio(self, portfolio_id: int) -> bool:
        """Delete a portfolio and all its data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Soft delete - mark as inactive
                cursor.execute('''
                    UPDATE portfolios SET status = 'inactive' WHERE id = ?
                ''', (portfolio_id,))
                
                conn.commit()
                logger.info(f"Deleted portfolio with ID: {portfolio_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting portfolio {portfolio_id}: {str(e)}")
            return False
    
    def get_portfolio_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count portfolios
                cursor.execute('SELECT COUNT(*) FROM portfolios WHERE status = "active"')
                portfolio_count = cursor.fetchone()[0]
                
                # Count holdings
                cursor.execute('''
                    SELECT COUNT(*) FROM holdings h
                    JOIN portfolios p ON h.portfolio_id = p.id
                    WHERE p.status = "active"
                ''')
                holdings_count = cursor.fetchone()[0]
                
                # Count transactions
                cursor.execute('''
                    SELECT COUNT(*) FROM transactions t
                    JOIN portfolios p ON t.portfolio_id = p.id
                    WHERE p.status = "active"
                ''')
                transactions_count = cursor.fetchone()[0]
                
                # Total portfolio value - try multiple fields
                cursor.execute('''
                    SELECT SUM(COALESCE(current_value, market_value, 0)) FROM holdings h
                    JOIN portfolios p ON h.portfolio_id = p.id
                    WHERE p.status = "active"
                ''')
                total_value = cursor.fetchone()[0] or 0
                
                # If still 0, try using market_value
                if total_value == 0:
                    cursor.execute('''
                        SELECT SUM(market_value) FROM holdings h
                        JOIN portfolios p ON h.portfolio_id = p.id
                        WHERE p.status = "active"
                    ''')
                    total_value = cursor.fetchone()[0] or 0
                
                # If still 0, try using cost_value as fallback
                if total_value == 0:
                    cursor.execute('''
                        SELECT SUM(cost_value) FROM holdings h
                        JOIN portfolios p ON h.portfolio_id = p.id
                        WHERE p.status = "active"
                    ''')
                    total_value = cursor.fetchone()[0] or 0
                
                return {
                    'portfolio_count': portfolio_count,
                    'holdings_count': holdings_count,
                    'transactions_count': transactions_count,
                    'total_value': total_value
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {str(e)}")
            return {'portfolio_count': 0, 'holdings_count': 0, 'transactions_count': 0, 'total_value': 0}

def calculate_xirr(cash_flows, dates, guess=0.1):
    """
    Calculate XIRR (Extended Internal Rate of Return) for a series of cash flows
    
    Args:
        cash_flows: List of cash flows (negative for investments, positive for returns)
        dates: List of corresponding dates
        guess: Initial guess for the rate (default 0.1 = 10%)
    
    Returns:
        XIRR as a decimal (e.g., 0.12 for 12%)
    """
    try:
        if len(cash_flows) != len(dates) or len(cash_flows) < 2:
            return None
        
        # Convert dates to datetime if they're not already
        if not isinstance(dates[0], datetime):
            dates = [pd.to_datetime(d) for d in dates]
        
        # Calculate days from first date
        first_date = dates[0]
        days_from_start = [(d - first_date).days for d in dates]
        
        # Define the NPV function
        def npv(rate):
            return sum(cf / (1 + rate) ** (days / 365.25) for cf, days in zip(cash_flows, days_from_start))
        
        # Use fsolve to find the rate where NPV = 0
        result = fsolve(npv, guess)
        
        if len(result) > 0 and not np.isnan(result[0]) and not np.isinf(result[0]):
            return result[0]
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating XIRR: {str(e)}")
        return None

def calculate_portfolio_xirr(transactions_df, current_value, current_date=None):
    """
    Calculate XIRR for a portfolio based on transactions and current value
    
    Args:
        transactions_df: DataFrame with transaction data
        current_value: Current portfolio value
        current_date: Current date (defaults to today)
    
    Returns:
        XIRR as a percentage
    """
    try:
        if current_date is None:
            current_date = datetime.now()
        
        if transactions_df.empty:
            return None
        
        # Prepare cash flows
        cash_flows = []
        dates = []
        
        # Add all transactions as cash flows
        for _, row in transactions_df.iterrows():
            if pd.notna(row.get('amount')) and pd.notna(row.get('date')):
                # Convert date to datetime if needed
                if not isinstance(row['date'], datetime):
                    trans_date = pd.to_datetime(row['date'])
                else:
                    trans_date = row['date']
                
                # Determine if it's an investment (negative) or withdrawal (positive)
                amount = float(row['amount'])
                if 'purchase' in str(row.get('description', '')).lower() or 'investment' in str(row.get('description', '')).lower():
                    cash_flows.append(-abs(amount))  # Investment (negative)
                elif 'redemption' in str(row.get('description', '')).lower() or 'withdrawal' in str(row.get('description', '')).lower():
                    cash_flows.append(abs(amount))   # Withdrawal (positive)
                else:
                    # Default: treat positive amounts as investments, negative as withdrawals
                    cash_flows.append(-amount)
                
                dates.append(trans_date)
        
        # Add current value as final cash flow (positive - what we would get if we sold today)
        if current_value > 0:
            cash_flows.append(current_value)
            dates.append(current_date)
        
        if len(cash_flows) < 2:
            return None
        
        # Calculate XIRR
        xirr_rate = calculate_xirr(cash_flows, dates)
        
        if xirr_rate is not None:
            return xirr_rate * 100  # Convert to percentage
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error calculating portfolio XIRR: {str(e)}")
        return None

def calculate_fund_xirr(fund_name: str, all_transactions_df: pd.DataFrame, holdings_df: pd.DataFrame) -> Optional[float]:
    """
    Calculate XIRR for a specific fund based on its transactions and current value.
    """
    try:
        fund_transactions_df = all_transactions_df[all_transactions_df['scheme'] == fund_name].copy()
        
        if fund_transactions_df.empty:
            return None
        
        # Get current value for this specific fund
        current_fund_holding = holdings_df[holdings_df['fund_name'] == fund_name]
        if current_fund_holding.empty:
            return None
        
        current_value = current_fund_holding['current_value'].sum()
        
        return calculate_portfolio_xirr(fund_transactions_df, current_value)
    except Exception as e:
        logger.error(f"Error calculating XIRR for fund {fund_name}: {str(e)}")
        return None

# Global variable to store live NAV data
LIVE_NAV_DATA = {}
LAST_NAV_FETCH_TIME = None
NAV_REFRESH_INTERVAL_HOURS = 6 # Refresh NAV data every 6 hours

def fetch_live_nav_data():
    """
    Fetches live NAV data from AMFI website (NAVAll.txt) and stores it in LIVE_NAV_DATA.

    This function is designed to be robust against small format variations in NAVAll.txt.
    For each line it attempts to extract:
      - scheme code / AMFI code (first token if numeric)
      - ISIN (token containing 'INF')
      - fund name (commonly token at index 1)
      - NAV (first numeric token found when scanning from the end)
      - date (if present in common formats)

    The resulting LIVE_NAV_DATA is keyed by multiple lookup keys to make subsequent
    matching more reliable: ISIN, normalized fund name, and scheme/amfi code (string).
    """
    global LIVE_NAV_DATA, LAST_NAV_FETCH_TIME

    # Use cached data for a short while to avoid spamming the AMFI endpoint
    if LAST_NAV_FETCH_TIME and (datetime.now() - LAST_NAV_FETCH_TIME) < timedelta(hours=NAV_REFRESH_INTERVAL_HOURS):
        logger.info("Using cached NAV data.")
        return

    logger.info("Fetching live NAV data from AMFI...")
    url = "https://portal.amfiindia.com/spages/NAVAll.txt"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text

        nav_data = {}

        # Helper to normalize fund names
        def normalize_name_for_key(name: str) -> str:
            if not name:
                return ""
            n = name.lower().strip()
            n = re.sub(r"\s*-\s*(regular|direct|plan|growth|dividend|reinvestment).*", "", n, flags=re.I)
            n = re.sub(r"\(.*\)", "", n)
            n = re.sub(r"[^a-z0-9 ]+", " ", n)
            n = re.sub(r"\s+", " ", n).strip()
            return n

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Skip obvious header lines
            if any(h in line for h in ("Scheme Code", "Scheme Name")) or line.lower().startswith('scheme'):
                continue

            # NAVAll.txt typically uses semicolons as delimiters
            parts = [p.strip() for p in re.split(r';', line) if p is not None]
            if not parts:
                continue

            scheme_code = None
            fund_name = None
            isin = None
            nav = None
            nav_date = None

            # Scheme code is often the first token and numeric
            first = parts[0] if len(parts) > 0 else ''
            if first and re.match(r'^\d+$', first):
                scheme_code = first

            # Collect ISINs (sometimes multiple ISIN columns); pick the first valid INF code
            isin_candidates = []
            for p in parts:
                if p and re.search(r'INF[A-Z0-9]{5,}', p, flags=re.I):
                    isin_candidates.append(re.search(r'INF[A-Z0-9]{5,}', p, flags=re.I).group(0).upper())
            if isin_candidates:
                isin = isin_candidates[0]

            # Identify NAV by scanning tokens from the end; capture its index to help infer scheme name
            nav_index = None
            for idx in range(len(parts) - 1, -1, -1):
                p = parts[idx]
                if not p:
                    continue
                p_clean = p.replace(',', '').replace('\u200b', '').strip()
                try:
                    nav_candidate = float(p_clean)
                    if 0 < nav_candidate < 1e8:
                        nav = nav_candidate
                        nav_index = idx
                        break
                except Exception:
                    # Avoid accidentally extracting date-day (e.g. '20' from '20-Oct-2025') as NAV.
                    # Only accept numeric substring when it looks like a real NAV: contains a decimal point
                    # or is a large integer (length > 3) or original token had commas (thousands sep).
                    num_match = re.search(r'\d+[.,]?\d*', p_clean)
                    if num_match:
                        try:
                            num_str = num_match.group(0).replace(',', '')
                            # Require either a decimal point or a sufficiently long number to consider it a NAV
                            if ('.' in num_str) or (len(num_str) > 3) or (',' in p):
                                nav_candidate = float(num_str)
                                if 0 < nav_candidate < 1e8:
                                    nav = nav_candidate
                                    nav_index = idx
                                    break
                        except Exception:
                            # ignore and continue searching
                            pass
                    continue

            # Attempt to find a date token (prefer token immediately after NAV if present)
            if nav_index is not None and nav_index + 1 < len(parts):
                p = parts[nav_index + 1]
                for fmt in ('%d-%b-%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%B-%Y'):
                    try:
                        nav_date = datetime.strptime(p, fmt)
                        break
                    except Exception:
                        continue

            # Infer fund/scheme name as tokens between scheme code (or start) and the NAV token, skipping ISIN-like tokens
            if nav_index is not None:
                name_start = 1 if scheme_code is not None else 0
                # Skip leading ISIN-like tokens
                while name_start < len(parts) and re.search(r'INF[A-Z0-9]{5,}', parts[name_start], flags=re.I):
                    name_start += 1

                name_end = nav_index - 1
                if name_end >= name_start:
                    candidate_tokens = []
                    for t in parts[name_start:name_end + 1]:
                        if not t or t == '-' or re.match(r'^\d+$', t):
                            continue
                        if re.search(r'INF[A-Z0-9]{5,}', t, flags=re.I):
                            continue
                        candidate_tokens.append(t)
                    if candidate_tokens:
                        fund_name = ' '.join(candidate_tokens).strip()

            # Fallbacks for fund_name
            if not fund_name:
                # Often scheme name is at index 3 in some files
                if len(parts) > 3 and parts[3]:
                    fund_name = parts[3]
                elif len(parts) > 1:
                    fund_name = parts[1]

            # If we have NAV and at least fund_name or ISIN or scheme_code, store the entry
            if nav is not None and (fund_name or isin or scheme_code):
                entry = {
                    'nav': nav,
                    'date': nav_date,
                    'fund_name': fund_name,
                    'isin': isin,
                    'amfi': scheme_code
                }

                # Store by ISIN key (preferred)
                if isin:
                    nav_data[isin] = entry

                # Store by normalized fund name
                if fund_name:
                    nav_data[normalize_name_for_key(fund_name)] = entry

                # Store by scheme/amfi code as string
                if scheme_code:
                    nav_data[str(scheme_code)] = entry


        LIVE_NAV_DATA = nav_data
        LAST_NAV_FETCH_TIME = datetime.now()
        logger.info(f"Successfully fetched {len(LIVE_NAV_DATA)} NAV entries.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching live NAV data: {e}")
        st.warning(f"Could not fetch live NAV data: {e}. Using cached NAVs if available.")
    except Exception as e:
        logger.error(f"Unexpected error while fetching NAV data: {e}")
        st.warning(f"Unexpected error while fetching NAV data: {e}. Using cached NAVs if available.")



def get_current_nav(fund_name: str, isin: str = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest NAV for a fund using LIVE_NAV_DATA.
    Returns a dict with keys: 'nav', 'date' and optionally 'isin' and 'fund_name'.
    If NAV cannot be found, returns None.

    Lookup strategy (in order):
      1. Exact ISIN match (case-insensitive)
      2. Exact AMFI/scheme code match (if fund row has 'amfi' code)
      3. Exact normalized fund name match
      4. Normalized substring match across stored fund names
      5. Partial ISIN match (last 6 chars)
    """
    global LIVE_NAV_DATA

    # Ensure live NAV data is available (attempt fetch if empty)
    if not LIVE_NAV_DATA:
        fetch_live_nav_data()

    def normalize_name(name: str) -> str:
        if not name:
            return ""
        n = name.lower().strip()
        n = re.sub(r"\s*-\s*(regular|direct|plan|growth|dividend|reinvestment).*", "", n, flags=re.I)
        n = re.sub(r"\(.*\)", "", n)
        n = re.sub(r"[^a-z0-9 ]+", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    # 1) Exact ISIN lookup (normalize to uppercase)
    if isin:
        isin_key = isin.strip().upper()
        if isin_key in LIVE_NAV_DATA:
            entry = LIVE_NAV_DATA[isin_key]
            return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # 2) Exact AMFI/scheme code lookup (some holdings have 'amfi' field)
    # Accept numeric or string AMFI codes
    try:
        possible_amfi = str(int(isin)) if isin and str(isin).isdigit() else None
    except Exception:
        possible_amfi = None

    if possible_amfi and possible_amfi in LIVE_NAV_DATA:
        entry = LIVE_NAV_DATA[possible_amfi]
        return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # If holdings passed AMFI code in fund_name accidentally, try that
    if fund_name and isinstance(fund_name, (str,)) and fund_name.isdigit() and fund_name in LIVE_NAV_DATA:
        entry = LIVE_NAV_DATA[fund_name]
        return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # 3) Direct lookup by exact normalized fund name
    if fund_name:
        fname_key = normalize_name(fund_name)
        if fname_key in LIVE_NAV_DATA:
            entry = LIVE_NAV_DATA[fname_key]
            return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # 4) Normalized substring matching - try to be forgiving with naming differences
    norm = normalize_name(fund_name or "")
    if norm:
        for key, entry in LIVE_NAV_DATA.items():
            if not isinstance(key, str):
                continue
            stored_name = (entry.get('fund_name') or key).lower()
            if norm in normalize_name(stored_name):
                return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # 5) Fallback: try partial ISIN match (last 6 characters)
    if isin:
        partial = isin[-6:]
        for key, entry in LIVE_NAV_DATA.items():
            k = str(key)
            if partial and partial in k:
                return {'nav': entry['nav'], 'date': entry.get('date'), 'isin': entry.get('isin'), 'fund_name': entry.get('fund_name')}

    # Not found
    logger.warning(f"NAV not found for '{fund_name}' (ISIN: {isin}).")
    return None



def safe_date_conversion(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """Safely convert date column to datetime format"""
    if df.empty or date_column not in df.columns:
        return df
    
    # Convert to datetime with error handling
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=[date_column])
    
    return df

def calculate_performance_metrics(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics for holdings using live NAVs from AMFI (NAVAll.txt).
    Ensures NAVs are fetched from get_current_nav (no mocking). Computes NAV change
    percent relative to parsed NAV if available and uses live NAV for current_value calculation.
    """
    if holdings_df.empty:
        return holdings_df

    def get_nav_and_change_for_row(row):
        try:
            entry = get_current_nav(row.get('fund_name', ''), row.get('isin', None))
            if entry:
                current_nav = float(entry.get('nav', 0.0) or 0.0)
                parsed_nav = float(row.get('nav', 0) or 0.0)
                nav_change_percent = 0.0
                if parsed_nav > 0:
                    try:
                        nav_change_percent = ((current_nav - parsed_nav) / parsed_nav) * 100
                    except Exception:
                        nav_change_percent = 0.0
                return {'nav': current_nav, 'change': round(nav_change_percent, 6), 'date': entry.get('date')}
        except Exception as e:
            logger.debug(f"Error fetching NAV for {row.get('fund_name')}: {e}")

        # Fallback to parsed NAV only (no mock values)
        parsed_nav = float(row.get('nav', 0) or 0.0)
        return {'nav': parsed_nav if parsed_nav > 0 else 0.0, 'change': 0.0, 'date': None}

    # Apply NAV lookup
    nav_series = holdings_df.apply(lambda r: get_nav_and_change_for_row(r), axis=1)
    holdings_df['current_nav'] = nav_series.apply(lambda x: x.get('nav', 0.0) if isinstance(x, dict) else 0.0)
    holdings_df['nav_change_percent'] = nav_series.apply(lambda x: x.get('change', 0.0) if isinstance(x, dict) else 0.0)

    # Calculate current value: prefer market_value if present and >0 else units * current_nav
    holdings_df['current_value'] = holdings_df.apply(
        lambda row: row['market_value'] if (pd.notna(row.get('market_value')) and row.get('market_value', 0) > 0) else (row.get('units', 0) * row.get('current_nav', 0)),
        axis=1
    )

    # Calculate gains and returns
    holdings_df['absolute_gain'] = holdings_df['current_value'] - holdings_df['cost_value']
    holdings_df['return_percentage'] = (
        (holdings_df['current_value'] - holdings_df['cost_value']) / holdings_df['cost_value'] * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    # Today's gain based on NAV change percent
    holdings_df['todays_gain_value'] = holdings_df['current_value'] * (holdings_df['nav_change_percent'] / 100)
    holdings_df['todays_gain_percent'] = holdings_df['nav_change_percent']

    # Portfolio weight
    total_value = holdings_df['current_value'].sum()
    if total_value > 0:
        holdings_df['portfolio_weight'] = (
            (holdings_df['current_value'] / total_value * 100)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .round(2)
        )
    else:
        holdings_df['portfolio_weight'] = 0.0

    return holdings_df


# Page configuration
st.set_page_config(
    page_title="CAMS Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .portfolio-summary {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class CAMSParser:
    def __init__(self, filename: str = "Unknown"):
        self.transactions = []
        self.holdings = []
        self.portfolio_summary = {}
        self.raw_data = None
        self.filename = filename
        self.processing_time = 0
        self.parse_status = "Not Started"
        self.error_message = None
        
    def parse_pdf(self, pdf_file: str, password: Optional[str] = None) -> bool:
        """Parse CAMS PDF using casparser library with progress tracking"""
        start_time = time.time()
        self.parse_status = "Starting"
        logger.info(f"Starting to parse PDF: {self.filename}")
        
        try:
            # Create progress container
            progress_container = st.container()
            with progress_container:
                st.info(f"🔄 Processing {self.filename}...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Reading PDF
                status_text.text("📖 Reading PDF file...")
                progress_bar.progress(10)
                logger.info(f"Reading PDF file: {pdf_file}")
                
                # Use casparser to read the PDF with JSON output
                json_str = casparser.read_cas_pdf(pdf_file, password, output="json")
                progress_bar.progress(30)
                
                # Parse JSON string to Python object
                import json
                self.raw_data = json.loads(json_str)
                
                # Step 2: Extracting data
                status_text.text("🔍 Extracting data from PDF...")
                progress_bar.progress(50)
                logger.info("Extracting data from casparser output")
                
                self._extract_data_from_casparser()
                progress_bar.progress(80)
                
                # Step 3: Finalizing
                status_text.text("✅ Finalizing data processing...")
                progress_bar.progress(100)
                
                self.processing_time = time.time() - start_time
                self.parse_status = "Success"
                
                status_text.text(f"✅ Successfully processed {self.filename} in {self.processing_time:.2f}s")
                logger.info(f"Successfully parsed {self.filename} in {self.processing_time:.2f} seconds")
                
                # Clear progress after a short delay
                time.sleep(1)
                progress_container.empty()
                
                return True
                
        except Exception as e:
            self.processing_time = time.time() - start_time
            self.parse_status = "Failed"
            self.error_message = str(e)
            
            error_msg = f"❌ Error parsing {self.filename}: {str(e)}"
            st.error(error_msg)
            logger.error(f"Error parsing {self.filename}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to basic parsing if casparser fails
            try:
                st.warning(f"🔄 Attempting fallback parsing for {self.filename}...")
                self._fallback_parsing(pdf_file)
                self.parse_status = "Fallback Success"
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed for {self.filename}: {str(fallback_error)}")
                return False
    
    def _extract_data_from_casparser(self):
        """Extract data from casparser output with detailed logging"""
        if not self.raw_data:
            logger.warning("No raw data available for extraction")
            return
            
        logger.info("Starting data extraction from casparser output")
        
        # Handle JSON data (dict) from casparser
        try:
            # raw_data is already a dict when using output="json"
            if isinstance(self.raw_data, dict):
                data_dict = self.raw_data
            else:
                # Fallback for other data types
                data_dict = dict(self.raw_data)
            
            logger.info(f"Raw data type: {type(self.raw_data)}")
            logger.info(f"Raw data attributes: {dir(self.raw_data)}")
            logger.info(f"Raw data keys: {list(data_dict.keys())}")
            
            # Save raw data to JSON for debugging
            import json
            with open(f"raw_data_{self.filename.replace('.pdf', '')}.json", 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved raw data to raw_data_{self.filename.replace('.pdf', '')}.json")
            
            # Extract portfolio summary
            if 'statement_period' in data_dict:
                self.portfolio_summary['statement_period'] = data_dict['statement_period']
                logger.info(f"Statement period: {self.portfolio_summary['statement_period']}")
            
            # Check for different possible data structures
            logger.info("Checking for different data structures...")
            
            # Method 1: Check for 'folios' key
            if 'folios' in data_dict:
                logger.info(f"Found 'folios' key with {len(data_dict['folios'])} folios")
                self._extract_from_folios(data_dict['folios'])
            
            # Method 2: Check for 'schemes' key directly
            elif 'schemes' in data_dict:
                logger.info(f"Found 'schemes' key with {len(data_dict['schemes'])} schemes")
                self._extract_from_schemes(data_dict['schemes'])
            
            # Method 3: Check for other possible keys
            else:
                logger.warning("No 'folios' or 'schemes' key found. Available keys:")
                for key in data_dict.keys():
                    logger.warning(f"  - {key}: {type(data_dict[key])}")
                    if isinstance(data_dict[key], (list, dict)):
                        logger.warning(f"    Content: {str(data_dict[key])[:200]}...")
            
            # Calculate portfolio totals
            total_market_value = sum(holding['market_value'] for holding in self.holdings)
            total_cost_value = sum(holding['cost_value'] for holding in self.holdings)
            
            self.portfolio_summary['total_market_value'] = total_market_value
            self.portfolio_summary['total_cost'] = total_cost_value
            
            logger.info(f"Data extraction completed:")
            logger.info(f"  - Holdings: {len(self.holdings)}")
            logger.info(f"  - Transactions: {len(self.transactions)}")
            logger.info(f"  - Total Market Value: Rs.{total_market_value:,.2f}")
            logger.info(f"  - Total Cost Value: Rs.{total_cost_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Try fallback extraction
            self._extract_any_available_data()
    
    def _extract_from_folios(self, folios):
        """Extract data from folios structure"""
        for i, folio in enumerate(folios):
            logger.info(f"Processing folio {i+1}/{len(folios)}: {folio.get('folio', 'Unknown')}")
            
            if 'schemes' in folio:
                logger.info(f"Found {len(folio['schemes'])} schemes in folio {folio.get('folio', 'Unknown')}")
                self._extract_from_schemes(folio['schemes'], folio)
            else:
                logger.warning(f"No 'schemes' key in folio {folio.get('folio', 'Unknown')}")
                logger.warning(f"Folio keys: {list(folio.keys())}")
    
    def _extract_from_schemes(self, schemes, folio_info=None):
        """Extract data from schemes structure"""
        for j, scheme in enumerate(schemes):
            logger.debug(f"Processing scheme {j+1}/{len(schemes)}: {scheme.get('scheme', 'Unknown')}")
            logger.debug(f"Scheme keys: {list(scheme.keys())}")
            
            # Extract current holdings - check for different possible structures
            current_units = 0
            current_value = 0
            cost_value = 0
            current_nav = 0
            
            # Method 1: Check for 'close' field (current balance)
            if 'close' in scheme and float(scheme['close']) > 0:
                current_units = float(scheme['close'])
                logger.info(f"Found close balance for scheme: {scheme.get('scheme', 'Unknown')} - {current_units} units")
            
            # Method 2: Check for 'valuation' field
            if 'valuation' in scheme and scheme['valuation']:
                valuation = scheme['valuation']
                current_value = float(valuation.get('value', 0))
                cost_value = float(valuation.get('cost', 0))
                current_nav = float(valuation.get('nav', 0))
                logger.info(f"Found valuation for scheme: {scheme.get('scheme', 'Unknown')} - Value: ₹{current_value}, Cost: ₹{cost_value}, NAV: {current_nav}")
            
            # Method 3: Check for 'current_balance' field (legacy)
            if 'current_balance' in scheme and scheme['current_balance']:
                current_balance = scheme['current_balance']
                current_units = float(current_balance.get('units', 0))
                current_value = float(current_balance.get('market_value', 0))
                cost_value = float(current_balance.get('cost_value', 0))
                current_nav = float(current_balance.get('nav', 0))
                logger.info(f"Found current_balance for scheme: {scheme.get('scheme', 'Unknown')}")
            
            # Add holding if we found any data
            if current_units > 0 or current_value > 0:
                holding = {
                    'fund_name': scheme.get('scheme', 'Unknown Fund'),
                    'isin': scheme.get('isin', ''),
                    'amfi': scheme.get('amfi', ''),
                    'rta': scheme.get('rta', ''),
                    'rta_code': scheme.get('rta_code', ''),
                    'advisor': scheme.get('advisor', ''),
                    'type': scheme.get('type', ''),
                    'units': current_units,
                    'market_value': current_value,
                    'cost_value': cost_value,
                    'nav': current_nav,
                    'folio': folio_info.get('folio', '') if folio_info else '',
                    'amc': folio_info.get('amc', '') if folio_info else '',
                    'pan': folio_info.get('PAN', '') if folio_info else '',
                    'kyc': folio_info.get('KYC', '') if folio_info else '',
                    'pankyc': folio_info.get('PANKYC', '') if folio_info else ''
                }
                self.holdings.append(holding)
                logger.info(f"Added holding: {holding['fund_name']} - Units: {current_units}, Value: ₹{current_value}")
            else:
                logger.warning(f"No current holdings found for scheme: {scheme.get('scheme', 'Unknown')}")
                logger.warning(f"Scheme keys: {list(scheme.keys())}")
                if 'close' in scheme:
                    logger.warning(f"Close value: {scheme['close']}")
                if 'valuation' in scheme:
                    logger.warning(f"Valuation: {scheme['valuation']}")
            
            # Extract transactions
            if 'transactions' in scheme:
                logger.info(f"Found {len(scheme['transactions'])} transactions for scheme: {scheme.get('scheme', 'Unknown')}")
                
                for k, transaction in enumerate(scheme['transactions']):
                    try:
                        # Try different date formats
                        date_str = transaction['date']
                        try:
                            # Try YYYY-MM-DD format first
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                        except ValueError:
                            try:
                                # Try DD-MMM-YYYY format
                                date = datetime.strptime(date_str, '%d-%b-%Y')
                            except ValueError:
                                try:
                                    # Try DD/MM/YYYY format
                                    date = datetime.strptime(date_str, '%d/%m/%Y')
                                except ValueError:
                                    # Default to current date if parsing fails
                                    date = datetime.now()
                                    logger.warning(f"Could not parse date '{date_str}', using current date")
                        
                        trans = {
                            'date': date,
                            'description': transaction['description'],
                            'amount': float(transaction['amount']),
                            'units': float(transaction['units']) if transaction['units'] is not None else 0,
                            'nav': float(transaction.get('nav', 0)) if transaction.get('nav') is not None else 0,
                            'balance': float(transaction.get('balance', 0)) if transaction.get('balance') is not None else 0,
                            'type': transaction.get('type', ''),
                            'scheme': scheme.get('scheme', ''),
                            'folio': folio_info.get('folio', '') if folio_info else ''
                        }
                        self.transactions.append(trans)
                        logger.debug(f"Added transaction: {trans['date']} - {trans['description']} - ₹{trans['amount']}")
                    except Exception as e:
                        logger.error(f"Error processing transaction {k+1}: {str(e)}")
                        logger.error(f"Transaction data: {transaction}")
                        continue
            else:
                logger.warning(f"No transactions found for scheme: {scheme.get('scheme', 'Unknown')}")
    
    def _fallback_parsing(self, pdf_file):
        """Fallback parsing method if casparser fails"""
        logger.warning(f"Using fallback parsing for {self.filename}")
        st.warning("Using fallback parsing method. Some features may not work correctly.")
        
        # Try to extract any available data from raw_data
        if self.raw_data:
            logger.info("Attempting to extract data from raw_data in fallback mode")
            self._extract_any_available_data()
        else:
            # Basic fallback - create minimal data structure
            self.holdings = []
            self.transactions = []
            self.portfolio_summary = {
                'total_market_value': 0,
                'total_cost': 0,
                'statement_period': 'Unknown'
            }
        
        logger.warning("Fallback parsing completed")
    
    def _extract_any_available_data(self):
        """Try to extract any available data from raw_data regardless of structure"""
        logger.info("Attempting to extract any available data from raw_data")
        
        # Reset data
        self.holdings = []
        self.transactions = []
        
        # Convert raw_data to dict if needed
        try:
            if isinstance(self.raw_data, dict):
                data_dict = self.raw_data
            else:
                data_dict = dict(self.raw_data)
        except:
            data_dict = self.raw_data
        
        # Try to find any data that looks like holdings or transactions
        def search_for_data(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Look for holdings-like data
                    if key.lower() in ['holdings', 'schemes', 'funds', 'investments']:
                        logger.info(f"Found potential holdings data at {current_path}")
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    self._try_extract_holding(item, current_path)
                    
                    # Look for transactions-like data
                    elif key.lower() in ['transactions', 'trades', 'activities']:
                        logger.info(f"Found potential transactions data at {current_path}")
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict):
                                    self._try_extract_transaction(item, current_path)
                    
                    # Recursively search
                    search_for_data(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_data(item, f"{path}[{i}]")
        
        search_for_data(data_dict)
        
        # Calculate totals
        total_market_value = sum(holding.get('market_value', 0) for holding in self.holdings)
        total_cost_value = sum(holding.get('cost_value', 0) for holding in self.holdings)
        
        # Try to get statement period safely
        statement_period = 'Unknown'
        try:
            if hasattr(self.raw_data, 'statement_period'):
                statement_period = self.raw_data.statement_period
            elif isinstance(data_dict, dict) and 'statement_period' in data_dict:
                statement_period = data_dict['statement_period']
        except:
            pass
        
        self.portfolio_summary = {
            'total_market_value': total_market_value,
            'total_cost': total_cost_value,
            'statement_period': statement_period
        }
        
        logger.info(f"Fallback extraction completed: {len(self.holdings)} holdings, {len(self.transactions)} transactions")
    
    def _try_extract_holding(self, data, path):
        """Try to extract holding data from any dictionary"""
        try:
            # Look for common field names
            fund_name = (data.get('scheme') or data.get('fund_name') or 
                        data.get('name') or data.get('fund') or 'Unknown Fund')
            
            market_value = 0
            cost_value = 0
            units = 0
            
            # Try different possible field names for values
            for value_field in ['market_value', 'current_value', 'value', 'amount']:
                if value_field in data:
                    market_value = float(data[value_field])
                    break
            
            for cost_field in ['cost_value', 'investment', 'cost', 'purchase_value']:
                if cost_field in data:
                    cost_value = float(data[cost_field])
                    break
            
            for units_field in ['units', 'quantity', 'balance']:
                if units_field in data:
                    units = float(data[units_field])
                    break
            
            if market_value > 0 or cost_value > 0:  # Only add if we found some value
                holding = {
                    'fund_name': fund_name,
                    'market_value': market_value,
                    'cost_value': cost_value,
                    'units': units,
                    'isin': data.get('isin', ''),
                    'folio': data.get('folio', ''),
                    'extracted_from': path
                }
                self.holdings.append(holding)
                logger.info(f"Extracted holding from {path}: {fund_name} - Rs.{market_value}")
        except Exception as e:
            logger.debug(f"Could not extract holding from {path}: {e}")
    
    def _try_extract_transaction(self, data, path):
        """Try to extract transaction data from any dictionary"""
        try:
            # Look for date field
            date_str = (data.get('date') or data.get('transaction_date') or 
                       data.get('date_of_transaction') or '01-Jan-2000')
            
            # Try to parse date
            try:
                date = datetime.strptime(date_str, '%d-%b-%Y')
            except:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    date = datetime(2000, 1, 1)  # Default date
            
            # Look for description
            description = (data.get('description') or data.get('transaction_type') or 
                          data.get('type') or data.get('narration') or 'Unknown Transaction')
            
            # Look for amount
            amount = 0
            for amount_field in ['amount', 'value', 'transaction_amount', 'net_amount']:
                if amount_field in data:
                    amount = float(data[amount_field])
                    break
            
            # Look for units
            units = 0
            for units_field in ['units', 'quantity', 'balance']:
                if units_field in data:
                    units = float(data[units_field])
                    break
            
            if amount != 0:  # Only add if we found an amount
                transaction = {
                    'date': date,
                    'description': description,
                    'amount': amount,
                    'units': units,
                    'extracted_from': path
                }
                self.transactions.append(transaction)
                logger.info(f"Extracted transaction from {path}: {date.strftime('%Y-%m-%d')} - {description} - ₹{amount}")
        except Exception as e:
            logger.debug(f"Could not extract transaction from {path}: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary for this parser"""
        return {
            'filename': self.filename,
            'status': self.parse_status,
            'processing_time': self.processing_time,
            'holdings_count': len(self.holdings),
            'transactions_count': len(self.transactions),
            'total_market_value': self.portfolio_summary.get('total_market_value', 0),
            'total_cost_value': self.portfolio_summary.get('total_cost', 0),
            'error_message': self.error_message
        }

def load_data_from_database(selected_portfolio=None):
    """Load portfolio data from database"""
    all_holdings = []
    all_transactions = []
    total_portfolio_value = 0
    total_cost_value = 0
    
    if selected_portfolio and selected_portfolio != "All Portfolios":
        # Load specific portfolio
        portfolio_index = None
        for i, p in enumerate(st.session_state.saved_portfolios):
            portfolio_display = f"{p['filename']} ({p['upload_date'][:10]})"
            if portfolio_display == selected_portfolio:
                portfolio_index = i
                break
        
        if portfolio_index is not None:
            portfolio = st.session_state.saved_portfolios[portfolio_index]
            all_holdings.extend(portfolio['holdings'])
            all_transactions.extend(portfolio['transactions'])
            total_portfolio_value = portfolio['portfolio_summary'].get('total_market_value', 0)
            total_cost_value = portfolio['portfolio_summary'].get('total_cost', 0)
    else:
        # Load all portfolios
        for portfolio in st.session_state.saved_portfolios:
            all_holdings.extend(portfolio['holdings'])
            all_transactions.extend(portfolio['transactions'])
            total_portfolio_value += portfolio['portfolio_summary'].get('total_market_value', 0)
            total_cost_value += portfolio['portfolio_summary'].get('total_cost', 0)
    
    return all_holdings, all_transactions, total_portfolio_value, total_cost_value

def create_dashboard(parsers=None, selected_portfolio=None):
    """Create the main dashboard with data from parsers or database"""
    
    # Clear any cached data when creating dashboard
    if 'cached_data' in st.session_state:
        del st.session_state.cached_data
    
    # Load data from database if no parsers provided or if viewing saved portfolios
    if not parsers or selected_portfolio:
        all_holdings, all_transactions, total_portfolio_value, total_cost_value = load_data_from_database(selected_portfolio)
    else:
        # Combine data from all parsers (for newly uploaded files)
        all_holdings = []
        all_transactions = []
        total_portfolio_value = 0
        total_cost_value = 0
        
        for parser in parsers:
            all_holdings.extend(parser.holdings)
            all_transactions.extend(parser.transactions)
            total_portfolio_value += parser.portfolio_summary.get('total_market_value', 0)
            total_cost_value += parser.portfolio_summary.get('total_cost', 0)
    
    # Create DataFrames
    holdings_df = pd.DataFrame(all_holdings)
    transactions_df = pd.DataFrame(all_transactions)
    
    # Calculate additional metrics for holdings
    if not holdings_df.empty:
        # Calculate performance metrics with current prices
        holdings_df = calculate_performance_metrics(holdings_df)
        
        # Extract fund house from fund name
        holdings_df['fund_house'] = holdings_df['fund_name'].apply(
            lambda x: x.split(' Mutual Fund')[0] if ' Mutual Fund' in x else 
                     x.split(' Fund')[0] if ' Fund' in x else 
                     x.split(' ')[0] if ' ' in x else x
        )
        
        # Calculate average purchase price
        holdings_df['avg_purchase_price'] = (
            holdings_df['cost_value'] / holdings_df['units']
        ).round(4)

    # Header
    st.markdown('<div class="main-header">📊 CAMS Portfolio Analyzer</div>', unsafe_allow_html=True)
    
    # Calculate current portfolio metrics
    if not holdings_df.empty:
        current_portfolio_value = holdings_df['current_value'].sum()
        total_invested = holdings_df['cost_value'].sum()
        total_gains = current_portfolio_value - total_invested
        returns_percent = (total_gains / total_invested * 100) if total_invested > 0 else 0
    else:
        current_portfolio_value = 0
        total_invested = 0
        total_gains = 0
        returns_percent = 0
    
    # Calculate XIRR
    xirr_percent = None
    if not transactions_df.empty and current_portfolio_value > 0:
        xirr_percent = calculate_portfolio_xirr(transactions_df, current_portfolio_value)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Current Portfolio Value",
            value=f"₹{current_portfolio_value:,.2f}",
            delta=f"₹{total_gains:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Total Returns",
            value=f"{returns_percent:.2f}%",
            delta=f"₹{total_gains:,.2f}"
        )
    
    with col3:
        st.metric(
            label="Amount Invested",
            value=f"₹{total_invested:,.2f}"
        )
    
    with col4:
        st.metric(
            label="Number of Funds",
            value=len(holdings_df) if not holdings_df.empty else 0
        )
    
    with col5:
        if xirr_percent is not None:
            st.metric(
                label="XIRR",
                value=f"{xirr_percent:.2f}%",
                delta="Annualized Return"
            )
        else:
            st.metric(
                label="XIRR",
                value="N/A",
                delta="Insufficient data"
            )
    
    # Individual Portfolio XIRR (when viewing all portfolios)
    if selected_portfolio == "All Portfolios" and st.session_state.saved_portfolios:
        st.subheader("📊 Individual Portfolio XIRR")
        
        portfolio_xirr_data = []
        for portfolio in st.session_state.saved_portfolios:
            portfolio_transactions = pd.DataFrame(portfolio['transactions'])
            portfolio_value = portfolio['portfolio_summary'].get('total_market_value', 0)
            
            if not portfolio_transactions.empty and portfolio_value > 0:
                portfolio_xirr = calculate_portfolio_xirr(portfolio_transactions, portfolio_value)
                portfolio_xirr_data.append({
                    'Portfolio': portfolio['filename'],
                    'Upload Date': portfolio['upload_date'][:10],
                    'Current Value': f"₹{portfolio_value:,.2f}",
                    'XIRR': f"{portfolio_xirr:.2f}%" if portfolio_xirr is not None else "N/A"
                })
            else:
                portfolio_xirr_data.append({
                    'Portfolio': portfolio['filename'],
                    'Upload Date': portfolio['upload_date'][:10],
                    'Current Value': f"₹{portfolio_value:,.2f}",
                    'XIRR': "N/A"
                })
        
        if portfolio_xirr_data:
            xirr_df = pd.DataFrame(portfolio_xirr_data)
            st.dataframe(xirr_df, use_container_width=True)
            
            # Download XIRR data
            csv = xirr_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_xirr.csv">📥 Download XIRR Data CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Portfolio Holdings", 
        "👥 Investor Details", 
        "📈 Portfolio Overview", 
        "🏦 Fund Allocation", 
        "📅 Transactions", 
        "📊 Performance", 
        "📋 Holdings Details"
    ])
    
    with tab1:
        display_portfolio_holdings_table(holdings_df, transactions_df)
    
    with tab2:
        display_investor_details(holdings_df)

    
    with tab3:
        display_portfolio_overview(holdings_df, current_portfolio_value, transactions_df, xirr_percent)
    
    with tab4:
        display_fund_allocation(holdings_df)
    
    with tab5:
        display_transactions(transactions_df)
    
    with tab6:
        display_performance_analysis(holdings_df, transactions_df)
    
    with tab7:
        display_holdings_details(holdings_df, transactions_df)
    
    # Add processing summary
    display_processing_summary(parsers)
    
    # Add extracted data preview
    display_extracted_data_preview(parsers)
    
    # Add additional information from casparser
    # if parsers and parsers[0].raw_data:
        # display_additional_info(parsers[0].raw_data)

def display_investor_details(holdings_df):
    """Display investor-wise investment details"""
    
    if holdings_df.empty:
        st.info("No holdings data available")
        return
    
    st.subheader("👥 Investor-Wise Investment Details")
    
    # Group by PAN to get investor-wise summary
    investor_summary = holdings_df.groupby('pan').agg({
        'folio': 'nunique',  # Number of folios per investor
        'fund_name': 'count',  # Number of funds per investor
        'cost_value': 'sum',  # Total investment per investor
        'current_value': 'sum',  # Current value per investor
        'absolute_gain': 'sum',  # Total gains per investor
        'return_percentage': 'mean',  # Average return per investor
        'amc': lambda x: ', '.join(x.unique())  # AMCs per investor
    }).round(2)
    
    # Calculate additional metrics
    investor_summary['total_gains'] = investor_summary['current_value'] - investor_summary['cost_value']
    investor_summary['overall_return_pct'] = (
        investor_summary['total_gains'] / investor_summary['cost_value'] * 100
    ).round(2)
    
    # Reset index to make PAN a column
    investor_summary = investor_summary.reset_index()
    
    # Calculate XIRR for each investor
    investor_xirr = {}
    
    # Aggregate all transactions from all saved portfolios
    all_transactions_list = []
    if 'saved_portfolios' in st.session_state and st.session_state.saved_portfolios:
        for portfolio in st.session_state.saved_portfolios:
            all_transactions_list.extend(portfolio['transactions'])
    
    all_transactions_df = pd.DataFrame(all_transactions_list)
    
    # Ensure transaction dates are datetime objects for filtering
    all_transactions_df = safe_date_conversion(all_transactions_df, 'date')
    
    for pan_number in investor_summary['pan'].unique():
        # Filter holdings for the current investor
        investor_holdings = holdings_df[holdings_df['pan'] == pan_number]
        
        # Get unique folios associated with this investor
        investor_folios = investor_holdings['folio'].unique()
        
        # Filter transactions for the current investor's folios
        investor_transactions = all_transactions_df[all_transactions_df['folio'].isin(investor_folios)]
        
        current_investor_value = investor_holdings['current_value'].sum()
        
        xirr = calculate_portfolio_xirr(investor_transactions, current_investor_value)
        investor_xirr[pan_number] = xirr
    
    investor_summary['xirr_percent'] = investor_summary['pan'].map(investor_xirr)
    
    # Rename columns for display
    display_columns = {
        'pan': 'PAN Number',
        'folio': 'Folios',
        'fund_name': 'Funds',
        'cost_value': 'Total Investment (₹)',
        'current_value': 'Current Value (₹)',
        'total_gains': 'Total Gains (₹)',
        'overall_return_pct': 'Overall Return (%)',
        'xirr_percent': 'XIRR (%)',
        'amc': 'Fund Houses'
    }
    
    display_df = investor_summary[list(display_columns.keys())].rename(columns=display_columns)
    
    # Format currency columns
    currency_cols = ['Total Investment (₹)', 'Current Value (₹)', 'Total Gains (₹)']
    for col in currency_cols:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
    
    # Format percentage column
    display_df['Overall Return (%)'] = display_df['Overall Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['XIRR (%)'] = display_df['XIRR (%)'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Add color coding for gains/losses
    def color_investor_gains(val):
        val_str = str(val)
        if 'Total Gains' in val_str or 'Overall Return' in val_str or 'XIRR' in val_str:
            if val_str.startswith('₹-') or val_str.startswith('-'):
                return 'color: red'
            elif val_str.startswith('₹') or val_str.startswith('+'):
                return 'color: green'
        return ''
    
    # Apply styling
    styled_df = display_df.style.applymap(color_investor_gains)
    
    # Display the table
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Investor-wise detailed breakdown
    st.subheader("📊 Detailed Investor Breakdown")
    
    # Create tabs for each investor
    investors = holdings_df['pan'].unique()
    if len(investors) > 1:
        investor_tabs = st.tabs([f"👤 {pan}" for pan in investors])
        
        for i, pan in enumerate(investors):
            with investor_tabs[i]:
                investor_data = holdings_df[holdings_df['pan'] == pan]
                
                # Investor summary metrics
                col1, col2, col3, col4, col5 = st.columns(5) # Added one more column for XIRR
                
                with col1:
                    st.metric(
                        "Total Investment",
                        f"₹{investor_data['cost_value'].sum():,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Current Value",
                        f"₹{investor_data['current_value'].sum():,.2f}"
                    )
                
                with col3:
                    total_gains = investor_data['current_value'].sum() - investor_data['cost_value'].sum()
                    st.metric(
                        "Total Gains",
                        f"₹{total_gains:,.2f}",
                        delta=f"{total_gains/investor_data['cost_value'].sum()*100:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "Number of Funds",
                        len(investor_data)
                    )
                
                with col5: # Display XIRR for individual investor
                    xirr_val = investor_xirr.get(pan)
                    if xirr_val is not None:
                        st.metric(
                            "XIRR",
                            f"{xirr_val:.2f}%",
                            delta="Annualized Return"
                        )
                    else:
                        st.metric(
                            "XIRR",
                            "N/A",
                            delta="Insufficient data"
                        )
                
                # Folio-wise breakdown for this investor
                st.subheader(f"📁 Folio-wise Breakdown for PAN: {pan}")
                
                folio_summary = investor_data.groupby('folio').agg({
                    'fund_name': 'count',
                    'cost_value': 'sum',
                    'current_value': 'sum',
                    'absolute_gain': 'sum',
                    'amc': lambda x: ', '.join(x.unique())
                }).round(2)
                
                folio_summary['folio_gains'] = folio_summary['current_value'] - folio_summary['cost_value']
                folio_summary['folio_return_pct'] = (
                    folio_summary['folio_gains'] / folio_summary['cost_value'] * 100
                ).round(2)
                
                folio_summary = folio_summary.reset_index()
                
                folio_display_columns = {
                    'folio': 'Folio Number',
                    'fund_name': 'Funds',
                    'cost_value': 'Investment (₹)',
                    'current_value': 'Current Value (₹)',
                    'folio_gains': 'Gains (₹)',
                    'folio_return_pct': 'Return (%)',
                    'amc': 'Fund Houses'
                }
                
                folio_display_df = folio_summary[list(folio_display_columns.keys())].rename(columns=folio_display_columns)
                
                # Format folio table
                folio_currency_cols = ['Investment (₹)', 'Current Value (₹)', 'Gains (₹)']
                for col in folio_currency_cols:
                    folio_display_df[col] = folio_display_df[col].apply(lambda x: f"₹{x:,.2f}")
                
                folio_display_df['Return (%)'] = folio_display_df['Return (%)'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(folio_display_df, width='stretch', hide_index=True)
                
                # Fund-wise breakdown for this investor
                st.subheader(f"📈 Fund-wise Performance for PAN: {pan}")
                
                fund_details = investor_data[['fund_name', 'folio', 'units', 'cost_value', 'current_value', 'absolute_gain', 'return_percentage', 'amc']].copy()
                fund_details = fund_details.sort_values('current_value', ascending=False)
                
                fund_display_columns = {
                    'fund_name': 'Fund Name',
                    'folio': 'Folio',
                    'units': 'Units',
                    'cost_value': 'Investment (₹)',
                    'current_value': 'Current Value (₹)',
                    'absolute_gain': 'Gains (₹)',
                    'return_percentage': 'Return (%)',
                    'amc': 'AMC'
                }
                
                fund_display_df = fund_details[list(fund_display_columns.keys())].rename(columns=fund_display_columns)
                
                # Format fund table
                fund_currency_cols = ['Investment (₹)', 'Current Value (₹)', 'Gains (₹)']
                for col in fund_currency_cols:
                    fund_display_df[col] = fund_display_df[col].apply(lambda x: f"₹{x:,.2f}")
                
                fund_display_df['Return (%)'] = fund_display_df['Return (%)'].apply(lambda x: f"{x:.2f}%")
                fund_display_df['Units'] = fund_display_df['Units'].apply(lambda x: f"{x:,.4f}")
                
                st.dataframe(fund_display_df, width='stretch', hide_index=True)
                
                # Export functionality for this investor
                csv = investor_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="investor_{pan}_holdings.csv">📥 Download {pan} Holdings CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    else:
        # Single investor - show detailed breakdown directly
        pan = investors[0]
        investor_data = holdings_df[holdings_df['pan'] == pan]
        
        st.info(f"Single investor detected: {pan}")
        
        # Show the same detailed breakdown as above but without tabs
        col1, col2, col3, col4, col5 = st.columns(5) # Added one more column for XIRR
        
        with col1:
            st.metric(
                "Total Investment",
                f"₹{investor_data['cost_value'].sum():,.2f}"
            )
        
        with col2:
            st.metric(
                "Current Value",
                f"₹{investor_data['current_value'].sum():,.2f}"
            )
        
        with col3:
            total_gains = investor_data['current_value'].sum() - investor_data['cost_value'].sum()
            st.metric(
                "Total Gains",
                f"₹{total_gains:,.2f}",
                delta=f"{total_gains/investor_data['cost_value'].sum()*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "Number of Funds",
                len(investor_data)
            )
        
        with col5: # Display XIRR for individual investor
            xirr_val = investor_xirr.get(pan)
            if xirr_val is not None:
                st.metric(
                    "XIRR",
                    f"{xirr_val:.2f}%",
                    delta="Annualized Return"
                )
            else:
                st.metric(
                    "XIRR",
                    "N/A",
                    delta="Insufficient data"
                )
        
        # Folio-wise breakdown
        st.subheader(f"📁 Folio-wise Breakdown for PAN: {pan}")
        
        folio_summary = investor_data.groupby('folio').agg({
            'fund_name': 'count',
            'cost_value': 'sum',
            'current_value': 'sum',
            'absolute_gain': 'sum',
            'amc': lambda x: ', '.join(x.unique())
        }).round(2)
        
        folio_summary['folio_gains'] = folio_summary['current_value'] - folio_summary['cost_value']
        folio_summary['folio_return_pct'] = (
            folio_summary['folio_gains'] / folio_summary['cost_value'] * 100
        ).round(2)
        
        folio_summary = folio_summary.reset_index()
        
        folio_display_columns = {
            'folio': 'Folio Number',
            'fund_name': 'Funds',
            'cost_value': 'Investment (₹)',
            'current_value': 'Current Value (₹)',
            'folio_gains': 'Gains (₹)',
            'folio_return_pct': 'Return (%)',
            'amc': 'Fund Houses'
        }
        
        folio_display_df = folio_summary[list(folio_display_columns.keys())].rename(columns=folio_display_columns)
        
        # Format folio table
        folio_currency_cols = ['Investment (₹)', 'Current Value (₹)', 'Gains (₹)']
        for col in folio_currency_cols:
            folio_display_df[col] = folio_display_df[col].apply(lambda x: f"₹{x:,.2f}")
        
        folio_display_df['Return (%)'] = folio_display_df['Return (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(folio_display_df, width='stretch', hide_index=True)
    
    # Export all investor data
    st.subheader("📥 Export Investor Data")
    csv = holdings_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="all_investors_holdings.csv">📥 Download All Investors Holdings CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def display_portfolio_holdings_table(holdings_df, transactions_df=None):
    """Display the main portfolio holdings table with current prices, returns and XIRR options.

    holdings_df: DataFrame of holdings
    transactions_df: DataFrame of all transactions (optional, used for XIRR calculations)
    """

    if holdings_df.empty:
        st.info("No holdings data available")
        return

    # Ensure transactions_df is a DataFrame if None
    if transactions_df is None:
        transactions_df = pd.DataFrame()

    # Safely convert transaction dates if provided
    if not transactions_df.empty and 'date' in transactions_df.columns:
        transactions_df = safe_date_conversion(transactions_df, 'date')

    st.subheader("📊 Portfolio Holdings")

    portfolio_table = holdings_df.copy()

    # Offer a mode: Raw Table or Pivot Table
    mode = st.radio("View Mode", ["Table", "Pivot (Group By)"])

    if mode == "Table":
        # Display the regular holdings table (compact)
        display_columns = {
            'fund_name': 'Mutual Fund Name',
            'pan': 'PAN Number',
            'units': 'Qty',
            'avg_purchase_price': 'Avg Purchase Price',
            'cost_value': 'Amount Invested',
            'current_nav': 'Current Price',
            'todays_gain_value': "Today's Gain (₹)",
            'todays_gain_percent': "Today's Gain (%)",
            'absolute_gain': 'Unrealized Gain (₹)',
            'return_percentage': 'Unrealized Gain (%)',
            'current_value': 'Current Value',
            'portfolio_weight': 'Portfolio Weight (%)'
        }

        cols_available = [c for c in display_columns.keys() if c in portfolio_table.columns]
        display_df = portfolio_table[cols_available].rename(columns={k: v for k, v in display_columns.items() if k in cols_available})

        # Optionally add Fund-level XIRR column when transactions available
        include_fund_xirr = True
        if not transactions_df.empty and 'scheme' in transactions_df.columns:
            include_fund_xirr = st.checkbox("Include Fund XIRR (per row)", value=True, help="Calculate XIRR per fund using transaction history. May take longer for many funds.")

        # Formatting helpers
        if 'avg_purchase_price' in display_df.columns:
            display_df['Avg Purchase Price'] = display_df['Avg Purchase Price'].apply(lambda x: f"₹{x:.4f}")
        if 'Amount Invested' in display_df.columns:
            display_df['Amount Invested'] = display_df['Amount Invested'].apply(lambda x: f"₹{x:,.2f}")
        if 'Current Price' in display_df.columns:
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"₹{x:.4f}")
        if "Today's Gain (₹)" in display_df.columns:
            display_df["Today's Gain (₹)"] = display_df["Today's Gain (₹)"].apply(lambda x: f"₹{x:,.2f}")
        if "Today's Gain (%)" in display_df.columns:
            display_df["Today's Gain (%)"] = display_df["Today's Gain (%)"].apply(lambda x: f"{x:.2f}%")
        if 'Unrealized Gain (₹)' in display_df.columns:
            display_df['Unrealized Gain (₹)'] = display_df['Unrealized Gain (₹)'].apply(lambda x: f"₹{x:,.2f}")
        if 'Unrealized Gain (%)' in display_df.columns:
            display_df['Unrealized Gain (%)'] = display_df['Unrealized Gain (%)'].apply(lambda x: f"{x:.2f}%")
        if 'Current Value' in display_df.columns:
            display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"₹{x:,.2f}")
        if 'Portfolio Weight (%)' in display_df.columns:
            display_df['Portfolio Weight (%)'] = display_df['Portfolio Weight (%)'].apply(lambda x: f"{x:.2f}%")

        # If user requested Fund XIRR, compute mapping (may be slow)
        if include_fund_xirr and 'fund_name' in portfolio_table.columns and not transactions_df.empty:
            st.info("Calculating XIRR for each fund. This may take a moment...")
            fund_xirr_map = {}
            unique_funds = portfolio_table['fund_name'].unique()
            for fund in unique_funds:
                try:
                    xirr_val = calculate_fund_xirr(fund, transactions_df, portfolio_table)
                    fund_xirr_map[fund] = xirr_val
                except Exception:
                    fund_xirr_map[fund] = None

            # Add column to display_df if fund_name present
            if 'Mutual Fund Name' in display_df.columns:
                display_df['Fund XIRR (%)'] = portfolio_table['fund_name'].map(lambda f: f"{fund_xirr_map.get(f):.2f}%" if pd.notna(fund_xirr_map.get(f)) else "N/A")

        # Color coding for gains/losses (simple)
        def color_gains(val):
            val_str = str(val)
            if '₹' in val_str and '-' in val_str:
                return 'color: red'
            if '₹' in val_str:
                return 'color: green'
            return ''

        try:
            styled_df = display_df.style.applymap(color_gains)
            st.dataframe(styled_df, width='stretch', hide_index=True)
        except Exception:
            st.dataframe(display_df, width='stretch', hide_index=True)

        # Summary metrics
        total_invested = portfolio_table.get('cost_value', pd.Series([0])).sum()
        total_current_value = portfolio_table.get('current_value', pd.Series([0])).sum()
        total_gains = total_current_value - total_invested
        total_gains_percent = (total_gains / total_invested * 100) if total_invested > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Invested", f"₹{total_invested:,.2f}")
        with col2:
            st.metric("Current Value", f"₹{total_current_value:,.2f}")
        with col3:
            st.metric("Total Gains", f"₹{total_gains:,.2f}", delta=f"{total_gains_percent:.2f}%")
        with col4:
            st.metric("Net Worth", f"₹{total_current_value:,.2f}")

        # Export
        csv = portfolio_table.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_holdings.csv">📥 Download Portfolio Holdings CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        # Pivot mode - allow grouping and aggregation
        st.info("Pivot Mode: Choose grouping columns and aggregation metrics")

        # Available groupby columns - prefer meaningful ones if present
        possible_group_cols = [c for c in ['fund_name', 'pan', 'folio', 'amc'] if c in portfolio_table.columns]
        if not possible_group_cols:
            st.warning("No suitable fields available for grouping (fund_name/pan/folio/amc not present).")
            return

        groupby_cols = st.multiselect("Group by (drag order matters)", possible_group_cols, default=['fund_name'])
        if not groupby_cols:
            st.warning("Select at least one grouping column")
            return

        # Numeric measure columns available for aggregation
        numeric_measures = [c for c in ['units', 'cost_value', 'current_value', 'absolute_gain', 'return_percentage', 'portfolio_weight'] if c in portfolio_table.columns]
        if not numeric_measures:
            st.warning("No numeric measure columns available to aggregate.")
            return

        measures = st.multiselect("Values to aggregate", numeric_measures, default=['current_value', 'cost_value', 'units'])
        agg_func = st.selectbox("Aggregation Function", ['sum', 'mean', 'median', 'max', 'min'], index=0)

        # Option to include XIRR per group
        include_xirr = True
        if not transactions_df.empty:
            include_xirr = st.checkbox("Include XIRR per group", value=True, help="Calculate XIRR for each pivot group using related transactions. May be slower.")

        # Perform groupby
        try:
            agg_map = {m: agg_func for m in measures}
            grouped = portfolio_table.groupby(groupby_cols).agg(agg_map)
            grouped = grouped.reset_index()
        except Exception as e:
            st.error(f"Failed to create pivot: {e}")
            return

        # If XIRR requested, compute for each grouped row
        if include_xirr and not transactions_df.empty:
            # Prebuild mapping folio -> pan or similar from holdings
            folio_pan_map = {}
            if 'folio' in portfolio_table.columns and 'pan' in portfolio_table.columns:
                folio_pan_map = portfolio_table.groupby('folio')['pan'].first().to_dict()

            xirr_list = []
            for _, row in grouped.iterrows():
                try:
                    # Build filter for holdings that belong to this group
                    mask = pd.Series([True] * len(portfolio_table))
                    for col in groupby_cols:
                        mask = mask & (portfolio_table[col] == row[col])

                    group_holdings = portfolio_table[mask]
                    current_value = float(group_holdings['current_value'].sum()) if 'current_value' in group_holdings.columns else 0

                    # Determine relevant transactions
                    relevant_txns = pd.DataFrame()

                    # If grouped by fund_name, prefer scheme-based filter
                    if 'fund_name' in groupby_cols and 'scheme' in transactions_df.columns:
                        fund_name = row.get('fund_name')
                        relevant_txns = transactions_df[transactions_df['scheme'] == fund_name]

                    # If grouped by folio, filter by folio
                    if relevant_txns.empty and 'folio' in groupby_cols and 'folio' in transactions_df.columns:
                        folio_val = row.get('folio')
                        relevant_txns = transactions_df[transactions_df['folio'] == folio_val]

                    # If grouped by pan, map pan -> folios and filter
                    if relevant_txns.empty and 'pan' in groupby_cols and 'folio' in transactions_df.columns:
                        pan_val = row.get('pan')
                        folios = portfolio_table[portfolio_table['pan'] == pan_val]['folio'].unique().tolist()
                        relevant_txns = transactions_df[transactions_df['folio'].isin(folios)]

                    # Fallback: use transactions that match any scheme present in group holdings (by fund_name)
                    if relevant_txns.empty and 'fund_name' in group_holdings.columns and 'scheme' in transactions_df.columns:
                        schemes = group_holdings['fund_name'].unique().tolist()
                        relevant_txns = transactions_df[transactions_df['scheme'].isin(schemes)]

                    # Compute xirr
                    if not relevant_txns.empty and current_value > 0:
                        xirr_val = calculate_portfolio_xirr(relevant_txns, current_value)
                        xirr_list.append(xirr_val)
                    else:
                        xirr_list.append(None)
                except Exception:
                    xirr_list.append(None)

            grouped['xirr_percent'] = xirr_list

        # Formatting grouped results for display
        display_grouped = grouped.copy()
        for col in ['cost_value', 'current_value', 'absolute_gain']:
            if col in display_grouped.columns:
                display_grouped[col] = display_grouped[col].apply(lambda x: f"₹{x:,.2f}")
        if 'return_percentage' in display_grouped.columns:
            display_grouped['return_percentage'] = display_grouped['return_percentage'].apply(lambda x: f"{x:.2f}%")
        if 'units' in display_grouped.columns:
            display_grouped['units'] = display_grouped['units'].apply(lambda x: f"{x:.4f}")
        if 'portfolio_weight' in display_grouped.columns:
            display_grouped['portfolio_weight'] = display_grouped['portfolio_weight'].apply(lambda x: f"{x:.2f}%")

        if 'xirr_percent' in display_grouped.columns:
            display_grouped['xirr_percent'] = display_grouped['xirr_percent'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

        st.dataframe(display_grouped, use_container_width=True)

        # Provide export for pivot
        csv = grouped.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_holdings_pivot.csv">📥 Download Pivot CSV</a>'
        st.markdown(href, unsafe_allow_html=True)



def display_portfolio_overview(holdings_df, total_value, transactions_df=None, xirr_percent=None):
    """Display portfolio overview with charts and metrics"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not holdings_df.empty:
            # Portfolio composition by fund house
            fig = px.pie(
                holdings_df, 
                values='current_value', 
                names='fund_house',
                title='Portfolio Allocation by Fund House',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if not holdings_df.empty:
            # Returns by fund - show top 10 by current value
            top_funds = holdings_df.nlargest(10, 'current_value')
            fig = px.bar(
                top_funds,
                x='fund_name',
                y='return_percentage',
                title='Top 10 Funds - Returns (%)',
                color='return_percentage',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch')
    
    # Portfolio metrics
    st.subheader("Portfolio Metrics")
    if not holdings_df.empty:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        
        with metrics_col1:
            top_fund = holdings_df.loc[holdings_df['current_value'].idxmax()]
            st.metric(
                "Largest Holding",
                f"₹{top_fund['current_value']:,.2f}",
                top_fund['fund_name'][:20] + "..."
            )
        
        with metrics_col2:
            best_performer = holdings_df.loc[holdings_df['return_percentage'].idxmax()]
            st.metric(
                "Best Performer",
                f"{best_performer['return_percentage']:.2f}%",
                best_performer['fund_name'][:20] + "..."
            )
        
        with metrics_col3:
            total_investment = holdings_df['cost_value'].sum()
            st.metric(
                "Total Investment",
                f"₹{total_investment:,.2f}"
            )
        
        with metrics_col4:
            st.metric(
                "Number of Fund Houses",
                f"{holdings_df['fund_house'].nunique()}"
            )
        
        with metrics_col5:
            if xirr_percent is not None:
                st.metric(
                    "XIRR (Annualized)",
                    f"{xirr_percent:.2f}%",
                    delta="Portfolio Return"
                )
            else:
                st.metric(
                    "XIRR (Annualized)",
                    "N/A",
                    delta="Insufficient data"
                )

def display_fund_allocation(holdings_df):
    """Display fund allocation analysis"""
    
    if holdings_df.empty:
        st.info("No holding data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current value by fund
        fig = px.treemap(
            holdings_df,
            path=['fund_house', 'fund_name'],
            values='current_value',
            title='Portfolio Treemap - Current Value'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Cost vs Current value comparison
        fig = px.scatter(
            holdings_df,
            x='cost_value',
            y='current_value',
            size='current_value',
            color='fund_house',
            hover_name='fund_name',
            title='Cost Value vs Current Value',
            labels={'cost_value': 'Cost Value (₹)', 'current_value': 'Current Value (₹)'}
        )
        st.plotly_chart(fig, width='stretch')

def display_transactions(transactions_df):
    """Display transaction history"""
    
    if transactions_df.empty:
        st.info("No transaction data available")
        return
    
    st.subheader("Transaction History")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    # Safely convert date column to datetime first
    transactions_df = safe_date_conversion(transactions_df, 'date')
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=transactions_df['date'].min().date() if not transactions_df.empty else datetime.now().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=transactions_df['date'].max().date() if not transactions_df.empty else datetime.now().date()
        )
    
    with col3:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["All", "Purchase", "Redemption", "SIP"]
        )
    
    # Filter transactions
    try:
        if not transactions_df.empty:
            filtered_df = transactions_df[
                (transactions_df['date'] >= pd.Timestamp(start_date)) &
                (transactions_df['date'] <= pd.Timestamp(end_date))
            ]
        else:
            filtered_df = transactions_df
    except Exception as e:
        st.warning(f"Error filtering transactions by date: {str(e)}")
        filtered_df = transactions_df
    
    if transaction_type != "All":
        filtered_df = filtered_df[
            filtered_df['description'].str.contains(transaction_type, case=False, na=False)
        ]
    
    # Display transactions
    st.dataframe(
        filtered_df.sort_values('date', ascending=False),
        width='stretch',
        hide_index=True
    )
    
    # Transaction statistics
    if not filtered_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_investment = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
            st.metric("Total Investment", f"₹{total_investment:,.2f}")
        
        with col2:
            total_redemption = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
            st.metric("Total Redemption", f"₹{total_redemption:,.2f}")
        
        with col3:
            net_flow = total_investment - total_redemption
            st.metric("Net Cash Flow", f"₹{net_flow:,.2f}")
        
        with col4:
            st.metric("Number of Transactions", len(filtered_df))

def display_performance_analysis(holdings_df, transactions_df):
    """Display performance analysis"""
    
    if holdings_df.empty:
        st.info("No data available for performance analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        fig = px.histogram(
            holdings_df,
            x='return_percentage',
            nbins=20,
            title='Distribution of Returns',
            labels={'return_percentage': 'Returns (%)'}
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Risk-Return scatter plot
        if len(holdings_df) > 1:
            # Calculate simple volatility (standard deviation of returns)
            avg_return = holdings_df['return_percentage'].mean()
            volatility = holdings_df['return_percentage'].std()
            
            fig = px.scatter(
                holdings_df,
                x=holdings_df['return_percentage'],
                y=[volatility] * len(holdings_df),  # Simplified volatility
                size='current_value',
                color='fund_house',
                hover_name='fund_name',
                title='Risk-Return Analysis',
                labels={'x': 'Returns (%)', 'y': 'Volatility'}
            )
            st.plotly_chart(fig, width='stretch')
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    if not holdings_df.empty:
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            avg_return = holdings_df['return_percentage'].mean()
            st.metric("Average Return", f"{avg_return:.2f}%")
        
        with metrics_col2:
            median_return = holdings_df['return_percentage'].median()
            st.metric("Median Return", f"{median_return:.2f}%")
        
        with metrics_col3:
            positive_returns = len(holdings_df[holdings_df['return_percentage'] > 0])
            st.metric("Funds in Profit", f"{positive_returns}/{len(holdings_df)}")
        
        with metrics_col4:
            max_drawdown = holdings_df['return_percentage'].min()
            st.metric("Worst Performing", f"{max_drawdown:.2f}%")

def display_holdings_details(holdings_df, transactions_df):
    """Display detailed holdings information"""
    
    if holdings_df.empty:
        st.info("No holdings data available")
        return
    
    st.subheader("Detailed Holdings")
    
    # Sortable dataframe
    sorted_df = holdings_df.sort_values('current_value', ascending=False)
    
    # Add additional metrics
    sorted_df['weight'] = (sorted_df['current_value'] / sorted_df['current_value'].sum()) * 100
    sorted_df['absolute_gain'] = sorted_df['current_value'] - sorted_df['cost_value']
    
    # Calculate fund-wise XIRR
    if not transactions_df.empty:
        fund_xirr_map = {}
        for fund_name in sorted_df['fund_name'].unique():
            xirr = calculate_fund_xirr(fund_name, transactions_df, sorted_df)
            fund_xirr_map[fund_name] = xirr
        sorted_df['fund_xirr'] = sorted_df['fund_name'].map(fund_xirr_map)
    else:
        sorted_df['fund_xirr'] = None
    
    # Display detailed table
    display_columns = {
        'fund_name': 'Fund Name',
        'fund_house': 'Fund House',
        'isin': 'ISIN',
        'rta': 'RTA',
        'type': 'Type',
        'units': 'Units',
        'current_nav': 'Current NAV',
        'cost_value': 'Cost Value (₹)',
        'current_value': 'Current Value (₹)',
        'absolute_gain': 'Absolute Gain (₹)',
        'return_percentage': 'Return (%)',
        'fund_xirr': 'Fund XIRR (%)',
        'weight': 'Portfolio Weight (%)'
    }
    
    display_df = sorted_df[list(display_columns.keys())].rename(columns=display_columns)
    
    # Format numbers
    for col in ['Cost Value (₹)', 'Current Value (₹)', 'Absolute Gain (₹)']:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
    
    for col in ['Return (%)', 'Portfolio Weight (%)']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    
    # Format current NAV
    display_df['Current NAV'] = display_df['Current NAV'].apply(lambda x: f"₹{x:.4f}")

    # Format Fund XIRR
    display_df['Fund XIRR (%)'] = display_df['Fund XIRR (%)'].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )
    
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Export option
    csv = sorted_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_holdings.csv">📥 Download Holdings CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def display_additional_info(raw_data):
    """Display additional information from casparser"""
    st.markdown("---")
    st.subheader("📋 Additional Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if isinstance(raw_data, dict) and 'investor_info' in raw_data:
            st.subheader("Investor Information")
            investor = raw_data['investor_info']
            st.write(f"**Name:** {investor.get('name', 'N/A')}")
            st.write(f"**Email:** {investor.get('email', 'N/A')}")
            st.write(f"**Mobile:** {investor.get('mobile', 'N/A')}")
            if investor.get('address'):
                st.write(f"**Address:** {investor['address']}")
    
    with col2:
        if isinstance(raw_data, dict) and 'statement_period' in raw_data:
            st.subheader("Statement Period")
            period = raw_data['statement_period']
            st.write(f"**From:** {period.get('from', 'N/A')}")
            st.write(f"**To:** {period.get('to', 'N/A')}")
        
        if isinstance(raw_data, dict) and 'file_type' in raw_data:
            st.write(f"**File Type:** {raw_data['file_type']}")
        
        if isinstance(raw_data, dict) and 'cas_type' in raw_data:
            st.write(f"**CAS Type:** {raw_data['cas_type']}")
    
    # Show folio summary
    if isinstance(raw_data, dict) and 'folios' in raw_data:
        st.subheader("Folio Summary")
        folio_summary = []
        for folio in raw_data['folios']:
            folio_info = {
                'Folio': folio.get('folio', 'N/A'),
                'AMC': folio.get('amc', 'N/A'),
                'PAN': folio.get('PAN', 'N/A'),
                'KYC': folio.get('KYC', 'N/A'),
                'PANKYC': folio.get('PANKYC', 'N/A'),
                'Schemes': len(folio.get('schemes', []))
            }
            folio_summary.append(folio_info)
        
        if folio_summary:
            folio_df = pd.DataFrame(folio_summary)
            st.dataframe(folio_df, width='stretch', hide_index=True)

def display_extracted_data_preview(parsers: List[CAMSParser]):
    """Display preview of extracted data to help debug parsing issues"""
    st.markdown("---")
    st.subheader("📊 Extracted Data Preview")
    
    if not parsers:
        st.info("No parsers available")
        return
    
    for parser in parsers:
        with st.expander(f"📁 {parser.filename} - Extracted Data"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Holdings Summary:**")
                if parser.holdings:
                    st.write(f"✅ Found {len(parser.holdings)} holdings")
                    for i, holding in enumerate(parser.holdings[:3]):  # Show first 3
                        st.write(f"{i+1}. {holding['fund_name']} - ₹{holding['market_value']:,.2f}")
                    if len(parser.holdings) > 3:
                        st.write(f"... and {len(parser.holdings) - 3} more")
                else:
                    st.write("❌ No holdings found")
                
                st.write("**Transactions Summary:**")
                if parser.transactions:
                    st.write(f"✅ Found {len(parser.transactions)} transactions")
                    for i, trans in enumerate(parser.transactions[:3]):  # Show first 3
                        st.write(f"{i+1}. {trans['date'].strftime('%Y-%m-%d')} - {trans['description']} - ₹{trans['amount']:,.2f}")
                    if len(parser.transactions) > 3:
                        st.write(f"... and {len(parser.transactions) - 3} more")
                else:
                    st.write("❌ No transactions found")
            
            with col2:
                st.write("**Portfolio Summary:**")
                st.write(f"Total Market Value: ₹{parser.portfolio_summary.get('total_market_value', 0):,.2f}")
                st.write(f"Total Cost Value: ₹{parser.portfolio_summary.get('total_cost', 0):,.2f}")
                
                if parser.raw_data:
                    st.write("**Raw Data Structure:**")
                    
                    # Convert raw_data to dict for display
                    try:
                        if isinstance(parser.raw_data, dict):
                            data_dict = parser.raw_data
                        else:
                            data_dict = dict(parser.raw_data)
                        
                        st.write(f"Type: {type(parser.raw_data)}")
                        st.write(f"Keys: {list(data_dict.keys())}")
                        
                        # Show specific data counts
                        if 'folios' in data_dict:
                            st.write(f"Folios: {len(data_dict['folios'])}")
                            for i, folio in enumerate(data_dict['folios'][:2]):
                                st.write(f"  Folio {i+1}: {folio.get('folio', 'Unknown')}")
                                if 'schemes' in folio:
                                    st.write(f"    Schemes: {len(folio['schemes'])}")
                        elif 'schemes' in data_dict:
                            st.write(f"Schemes: {len(data_dict['schemes'])}")
                        else:
                            st.write("⚠️ No folios or schemes found in raw data")
                            st.write("Available keys and their types:")
                            for key, value in data_dict.items():
                                st.write(f"  - {key}: {type(value)}")
                                if isinstance(value, (list, dict)) and len(str(value)) < 200:
                                    st.write(f"    Content: {value}")
                                elif isinstance(value, (list, dict)):
                                    st.write(f"    Content: {str(value)[:100]}...")
                    except Exception as e:
                        st.write(f"Error converting raw data: {str(e)}")
                        st.write(f"Raw data type: {type(parser.raw_data)}")
                        st.write(f"Raw data attributes: {dir(parser.raw_data)}")
                else:
                    st.write("❌ No raw data available")
            
            # Show detailed holdings if available
            if parser.holdings:
                st.write("**Detailed Holdings:**")
                holdings_df = pd.DataFrame(parser.holdings)
                st.dataframe(holdings_df[['fund_name', 'market_value', 'cost_value', 'units']], 
                           width='stretch', hide_index=True)
            
            # Show detailed transactions if available
            if parser.transactions:
                st.write("**Detailed Transactions:**")
                transactions_df = pd.DataFrame(parser.transactions)
                st.dataframe(transactions_df[['date', 'description', 'amount', 'units']], 
                           width='stretch', hide_index=True)

def display_processing_summary(parsers: List[CAMSParser]):
    """Display processing summary for all parsers"""
    st.markdown("---")
    st.subheader("📊 Processing Summary")
    
    if not parsers:
        st.info("No files processed")
        return
    
    # Create summary data
    summary_data = []
    total_processing_time = 0
    successful_files = 0
    failed_files = 0
    
    for parser in parsers:
        summary = parser.get_processing_summary()
        summary_data.append(summary)
        total_processing_time += summary['processing_time']
        
        if summary['status'] in ['Success', 'Fallback Success']:
            successful_files += 1
        else:
            failed_files += 1
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", len(parsers))
    
    with col2:
        st.metric("Successful", successful_files, delta=f"{successful_files/len(parsers)*100:.1f}%")
    
    with col3:
        st.metric("Failed", failed_files, delta=f"{failed_files/len(parsers)*100:.1f}%")
    
    with col4:
        st.metric("Total Processing Time", f"{total_processing_time:.2f}s")
    
    # Detailed processing table
    st.subheader("📋 File Processing Details")
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format the dataframe for display
    display_columns = {
        'filename': 'File Name',
        'status': 'Status',
        'processing_time': 'Processing Time (s)',
        'holdings_count': 'Holdings',
        'transactions_count': 'Transactions',
        'total_market_value': 'Market Value (₹)',
        'total_cost_value': 'Cost Value (₹)'
    }
    
    display_df = summary_df[list(display_columns.keys())].rename(columns=display_columns)
    
    # Format numbers
    display_df['Market Value (₹)'] = display_df['Market Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
    display_df['Cost Value (₹)'] = display_df['Cost Value (₹)'].apply(lambda x: f"₹{x:,.2f}")
    display_df['Processing Time (s)'] = display_df['Processing Time (s)'].apply(lambda x: f"{x:.2f}s")
    
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Show error details if any
    failed_parsers = [p for p in parsers if p.parse_status == 'Failed']
    if failed_parsers:
        st.subheader("❌ Error Details")
        for parser in failed_parsers:
            with st.expander(f"Error in {parser.filename}"):
                st.error(f"**Error:** {parser.error_message}")
                st.code(f"Status: {parser.parse_status}\nProcessing Time: {parser.processing_time:.2f}s")
    
    # Show debug information
    if st.checkbox("Show Debug Information"):
        st.subheader("🔍 Debug Information")
        
        for parser in parsers:
            with st.expander(f"Debug: {parser.filename}"):
                st.json(parser.get_processing_summary())
                
                if parser.raw_data:
                    # Convert raw_data to dict for display
                    try:
                        if isinstance(parser.raw_data, dict):
                            data_dict = parser.raw_data
                        else:
                            data_dict = dict(parser.raw_data)
                        
                        st.write("**Raw Data Type:**")
                        st.write(f"{type(parser.raw_data)}")
                        
                        st.write("**Raw Data Keys:**")
                        st.write(list(data_dict.keys()))
                        
                        # Show complete raw data structure
                        st.write("**Complete Raw Data Structure:**")
                        st.json(data_dict)
                        
                        if 'folios' in data_dict:
                            st.write(f"**Folios Count:** {len(data_dict['folios'])}")
                            
                            for i, folio in enumerate(data_dict['folios'][:2]):  # Show first 2 folios
                                st.write(f"**Folio {i+1}:** {folio.get('folio', 'Unknown')}")
                                st.write(f"Folio keys: {list(folio.keys())}")
                                if 'schemes' in folio:
                                    st.write(f"  - Schemes: {len(folio['schemes'])}")
                                    for j, scheme in enumerate(folio['schemes'][:3]):  # Show first 3 schemes
                                        st.write(f"    - Scheme {j+1}: {scheme.get('scheme', 'Unknown')}")
                                        st.write(f"    - Scheme keys: {list(scheme.keys())}")
                                        if 'current_balance' in scheme:
                                            st.write(f"    - Current Balance: {scheme['current_balance']}")
                                        if 'transactions' in scheme:
                                            st.write(f"      - Transactions: {len(scheme['transactions'])}")
                                            for k, trans in enumerate(scheme['transactions'][:2]):  # Show first 2 transactions
                                                st.write(f"        - Transaction {k+1}: {trans}")
                        else:
                            st.write("**No 'folios' key found. Available keys:**")
                            for key, value in data_dict.items():
                                st.write(f"- {key}: {type(value)}")
                                if isinstance(value, (list, dict)):
                                    st.write(f"  Content: {str(value)[:500]}...")
                    except Exception as e:
                        st.write(f"Error converting raw data: {str(e)}")
                        st.write(f"Raw data type: {type(parser.raw_data)}")
                        st.write(f"Raw data attributes: {dir(parser.raw_data)}")
    
    # Add raw data download option
    if st.checkbox("📥 Download Raw Data"):
        st.subheader("📥 Raw Data Files")
        
        for parser in parsers:
            if parser.raw_data:
                import json
                try:
                    # Convert raw_data to dict for JSON serialization
                    if isinstance(parser.raw_data, dict):
                        data_dict = parser.raw_data
                    else:
                        data_dict = dict(parser.raw_data)
                    
                    raw_data_json = json.dumps(data_dict, indent=2, default=str, ensure_ascii=False)
                    
                    st.download_button(
                        label=f"Download {parser.filename} Raw Data",
                        data=raw_data_json,
                        file_name=f"raw_data_{parser.filename.replace('.pdf', '')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error serializing raw data for {parser.filename}: {str(e)}")
    
    # Log processing summary
    logger.info("Processing Summary:")
    logger.info(f"  - Total files: {len(parsers)}")
    logger.info(f"  - Successful: {successful_files}")
    logger.info(f"  - Failed: {failed_files}")
    logger.info(f"  - Total processing time: {total_processing_time:.2f}s")

def main():
        # Fetch live NAV data on startup
    fetch_live_nav_data()
    
    # Password protection (can be enabled/disabled via env var PASSWORD_LOGIN_ENABLED)
    # Set PASSWORD_LOGIN_ENABLED to 'false', '0' or 'no' to disable password login
    PASSWORD_LOGIN_ENABLED = os.environ.get("PASSWORD_LOGIN_ENABLED", "true").strip().lower() in ("1", "true", "yes", "y")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not PASSWORD_LOGIN_ENABLED:
        # If password login is disabled via env, mark the session as logged in
        st.session_state.logged_in = True
        st.sidebar.info("Password login is disabled (PASSWORD_LOGIN_ENABLED=false).")
    else:
        # Normal password-protected flow
        if not st.session_state.logged_in:
            st.sidebar.title("Login")
            password_placeholder = st.sidebar.empty()
            password_input = password_placeholder.text_input("Enter Password", type="password", key="app_password")
            
            # Get password from environment variable
            APP_PASSWORD = os.environ.get("APP_PASSWORD", "Changeme123!")

            if st.sidebar.button("Login"):
                if APP_PASSWORD and password_input == APP_PASSWORD:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect password")
            st.stop() # Stop execution if not logged in


    st.sidebar.title("CAMS Portfolio Analyzer")
    st.sidebar.markdown("---")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Load existing portfolios from database
    if 'saved_portfolios' not in st.session_state:
        st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
    
    # Initialize selected_portfolio if not set
    if 'selected_portfolio' not in st.session_state:
        st.session_state.selected_portfolio = "All Portfolios"
    
    # Initialize processing_files flag if not set
    if 'processing_files' not in st.session_state:
        st.session_state.processing_files = False
    
    # Add refresh button for database data
    if st.sidebar.button("🔄 Refresh Database", key="refresh_db"):
        st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
        st.session_state.selected_portfolio = "All Portfolios"  # Reset selection
        st.success("Database refreshed!")
        st.rerun()
    
    # Add recalculate button for total value
    if st.sidebar.button("💰 Recalculate Total Value", key="recalc_value"):
        try:
            # This will trigger a refresh which will recalculate the stats
            st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
            st.success("Total value recalculated!")
            st.rerun()
        except Exception as e:
            st.error(f"Error recalculating: {str(e)}")
    
    # Database stats
    db_stats = st.session_state.db_manager.get_portfolio_stats()
    st.sidebar.metric("📊 Portfolios", db_stats['portfolio_count'])
    st.sidebar.metric("💰 Total Value", f"₹{db_stats['total_value']:,.2f}")
    
    # Debug info (remove in production)
    if st.sidebar.checkbox("🔍 Debug Database", key="debug_db"):
        st.sidebar.write(f"**Debug Info:**")
        st.sidebar.write(f"- Portfolios: {db_stats['portfolio_count']}")
        st.sidebar.write(f"- Holdings: {db_stats['holdings_count']}")
        st.sidebar.write(f"- Transactions: {db_stats['transactions_count']}")
        st.sidebar.write(f"- Total Value: {db_stats['total_value']}")
        
        # Show sample holdings data
        if st.session_state.saved_portfolios:
            sample_portfolio = st.session_state.saved_portfolios[0]
            if sample_portfolio['holdings']:
                sample_holding = sample_portfolio['holdings'][0]
                st.sidebar.write(f"**Sample Holding:**")
                st.sidebar.write(f"- Fund: {sample_holding.get('fund_name', 'N/A')}")
                st.sidebar.write(f"- Market Value: {sample_holding.get('market_value', 0)}")
                st.sidebar.write(f"- Current Value: {sample_holding.get('current_value', 0)}")
                st.sidebar.write(f"- Units: {sample_holding.get('units', 0)}")
    
    # Database status indicator
    if db_stats['portfolio_count'] > 0:
        st.sidebar.success(f"✅ Database: {db_stats['portfolio_count']} portfolios loaded")
    else:
        st.sidebar.info("📭 Database: No portfolios found")
    
    # Portfolio management section
    st.sidebar.subheader("📁 Portfolio Management")
    
    # Only show portfolio selection if not processing files
    if not st.session_state.get('processing_files', False):
        if st.session_state.saved_portfolios:
            # Show existing portfolios
            portfolio_options = [f"{p['filename']} ({p['upload_date'][:10]})" for p in st.session_state.saved_portfolios]
            selected_portfolio = st.sidebar.selectbox(
                "Select Portfolio to View:",
                ["All Portfolios"] + portfolio_options,
                key="portfolio_selector"
            )
            
            # Store selected portfolio in session state
            st.session_state.selected_portfolio = selected_portfolio
    else:
        # Show message when portfolio selection is hidden during processing
        st.sidebar.info("🔄 Processing files... Portfolio selection temporarily hidden.")
    
    # Portfolio actions (only show when not processing and portfolio is selected)
    if (not st.session_state.get('processing_files', False) and 
        st.session_state.saved_portfolios and 
        'selected_portfolio' in st.session_state and 
        st.session_state.selected_portfolio != "All Portfolios"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🗑️ Delete", key="delete_portfolio"):
                portfolio_index = portfolio_options.index(selected_portfolio)
                portfolio_id = st.session_state.saved_portfolios[portfolio_index]['id']
                if st.session_state.db_manager.delete_portfolio(portfolio_id):
                    st.success("Portfolio deleted successfully!")
                    st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", key="refresh_portfolio"):
                st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
                st.rerun()
    
    # Reset and export options (always available when portfolios exist)
    if st.session_state.saved_portfolios and not st.session_state.get('processing_files', False):
        # Reset all portfolios option
        if st.sidebar.button("🗑️ Reset All Portfolios", type="secondary"):
            if st.sidebar.checkbox("Confirm Reset", key="confirm_reset"):
                # Delete all portfolios
                for portfolio in st.session_state.saved_portfolios:
                    st.session_state.db_manager.delete_portfolio(portfolio['id'])
                st.session_state.saved_portfolios = []
                st.success("All portfolios reset successfully!")
                st.rerun()
        
        # Export database option
        st.sidebar.subheader("📤 Export Options")
        if st.sidebar.button("📥 Export Database", key="export_db"):
            # Create a combined CSV of all portfolios
            all_holdings = []
            all_transactions = []
            
            for portfolio in st.session_state.saved_portfolios:
                for holding in portfolio['holdings']:
                    holding['portfolio_name'] = portfolio['filename']
                    all_holdings.append(holding)
                
                for transaction in portfolio['transactions']:
                    transaction['portfolio_name'] = portfolio['filename']
                    all_transactions.append(transaction)
            
            # Export holdings
            if all_holdings:
                holdings_df = pd.DataFrame(all_holdings)
                csv_holdings = holdings_df.to_csv(index=False)
                b64_holdings = base64.b64encode(csv_holdings.encode()).decode()
                href_holdings = f'<a href="data:file/csv;base64,{b64_holdings}" download="all_portfolios_holdings.csv">📥 Download All Holdings CSV</a>'
                st.sidebar.markdown(href_holdings, unsafe_allow_html=True)
            
            # Export transactions
            if all_transactions:
                transactions_df = pd.DataFrame(all_transactions)
                csv_transactions = transactions_df.to_csv(index=False)
                b64_transactions = base64.b64encode(csv_transactions.encode()).decode()
                href_transactions = f'<a href="data:file/csv;base64,{b64_transactions}" download="all_portfolios_transactions.csv">📥 Download All Transactions CSV</a>'
                st.sidebar.markdown(href_transactions, unsafe_allow_html=True)
    elif not st.session_state.saved_portfolios:
        st.sidebar.info("No portfolios found. Upload PDF files to get started.")
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.subheader("📤 Upload New Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CAMS PDF Files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload your CAMS Consolidated Account Statement PDF files"
    )
    
    # Password input
    password = st.sidebar.text_input(
        "PDF Password (if required)",
        type="password",
        help="Enter the password for your CAMS PDF files if they are password protected"
    )
    
    # Auto-save option
    auto_save = st.sidebar.checkbox(
        "💾 Auto-save to Database",
        value=True,
        help="Automatically save processed portfolios to local database"
    )
    
    # Database reset option
    reset_database = st.sidebar.checkbox(
        "🗑️ Reset Database on Upload",
        value=False,
        help="Clear all existing portfolios before uploading new files"
    )
    
    st.sidebar.markdown("---")
    
    # Log viewer
    if st.sidebar.checkbox("📋 View Processing Logs"):
        st.sidebar.subheader("Processing Logs")
        
        # Read log file if it exists
        if os.path.exists('cams_analyzer.log'):
            with open('cams_analyzer.log', 'r') as f:
                log_content = f.read()
            
            # Show last 20 lines
            log_lines = log_content.split('\n')
            recent_logs = log_lines[-20:] if len(log_lines) > 20 else log_lines
            
            st.sidebar.text_area(
                "Recent Log Entries",
                value='\n'.join(recent_logs),
                height=200,
                help="Last 20 log entries"
            )
            
            if st.sidebar.button("Download Full Log"):
                st.sidebar.download_button(
                    label="📥 Download Log File",
                    data=log_content,
                    file_name=f"cams_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                    mime="text/plain"
                )
        else:
            st.sidebar.info("No log file found yet")
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        1. Upload your CAMS PDF files
        2. Enter password if required
        3. Wait for the files to be processed
        4. View portfolio summary and analytics
        5. Download detailed reports
        
        **Supported Formats:** CAMS Consolidated Account Statements
        """
    )
    
    # Check if user wants to view saved portfolio
    if (st.session_state.saved_portfolios and 
        'selected_portfolio' in st.session_state and 
        st.session_state.selected_portfolio != "None" and 
        st.session_state.selected_portfolio != "All Portfolios"):
        # Display saved portfolio
        portfolio_options = [f"{p['filename']} ({p['upload_date'][:10]})" for p in st.session_state.saved_portfolios]
        if st.session_state.selected_portfolio in portfolio_options:
            selected_idx = portfolio_options.index(st.session_state.selected_portfolio)
            selected_portfolio_data = st.session_state.saved_portfolios[selected_idx]
        else:
            # Portfolio not found, reset selection
            st.session_state.selected_portfolio = "All Portfolios"
            selected_portfolio_data = None
        
        # Convert saved data to parser objects for dashboard
        if selected_portfolio_data:
            parsers = []
            parser = CAMSParser(selected_portfolio_data['filename'])
            parser.holdings = selected_portfolio_data['holdings']
            parser.transactions = selected_portfolio_data['transactions']
            parser.portfolio_summary = selected_portfolio_data['portfolio_summary']
            parser.parse_status = "Loaded from Database"
            parser.processing_time = selected_portfolio_data['portfolio_summary'].get('processing_time', 0)
            parsers.append(parser)
            
            st.success(f"📁 Loaded portfolio: {selected_portfolio_data['filename']}")
            create_dashboard(parsers)
        else:
            # Show all portfolios
            create_dashboard(selected_portfolio="All Portfolios")
    
    elif st.session_state.saved_portfolios:
        # Show all saved portfolios
        create_dashboard(selected_portfolio="All Portfolios")
    
    elif uploaded_files:
        # Set processing flag to hide portfolio selection
        st.session_state.processing_files = True
        
        # Always reset portfolio selection when new files are uploaded
        st.session_state.selected_portfolio = "All Portfolios"
        
        # Reset database if requested
        if reset_database:
            st.info("🗑️ Resetting database as requested...")
            # Get current portfolios before clearing session state
            current_portfolios = st.session_state.saved_portfolios.copy() if 'saved_portfolios' in st.session_state else []
            for portfolio in current_portfolios:
                st.session_state.db_manager.delete_portfolio(portfolio['id'])
            st.session_state.saved_portfolios = []
            st.success("✅ Database reset completed!")
        
        # Clear all existing data from session state to force fresh start
        keys_to_clear = ['saved_portfolios', 'selected_portfolio']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reinitialize with fresh data
        st.session_state.selected_portfolio = "All Portfolios"
        
        # Show message that data is being reset
        st.info("🔄 Processing new files... Previous data will be replaced with new data.")
        
        parsers = []
        processing_start_time = time.time()
        
        logger.info(f"Starting to process {len(uploaded_files)} PDF files")
        
        # Create overall progress tracking
        st.subheader("📁 Processing Files")
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                overall_status.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                overall_progress.progress((i) / len(uploaded_files))
                
                logger.info(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Create parser with filename
                parser = CAMSParser(uploaded_file.name)
                
                # Save uploaded file temporarily
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                logger.info(f"Saved temporary file: {temp_filename}")
                
                # Parse using casparser with progress tracking
                success = parser.parse_pdf(temp_filename, password if password else None)
                
                if success:
                    parsers.append(parser)
                    logger.info(f"Successfully processed: {uploaded_file.name}")
                    
                    # Save to database if auto-save is enabled
                    if auto_save:
                        try:
                            file_content = uploaded_file.getvalue()
                            portfolio_id = st.session_state.db_manager.save_portfolio(
                                uploaded_file.name, file_content, parser
                            )
                            st.success(f"💾 Saved to database: {uploaded_file.name}")
                            
                            # Refresh saved portfolios
                            st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
                            
                        except Exception as save_error:
                            st.warning(f"⚠️ Failed to save {uploaded_file.name} to database: {str(save_error)}")
                            logger.error(f"Database save error for {uploaded_file.name}: {str(save_error)}")
                else:
                    logger.error(f"Failed to process: {uploaded_file.name}")
                
                # Clean up temporary file
                try:
                    os.remove(temp_filename)
                    logger.info(f"Cleaned up temporary file: {temp_filename}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up {temp_filename}: {cleanup_error}")
                    
            except Exception as e:
                error_msg = f"Unexpected error processing {uploaded_file.name}: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Complete overall progress
        overall_progress.progress(1.0)
        overall_status.text(f"✅ Completed processing {len(uploaded_files)} files")
        
        total_processing_time = time.time() - processing_start_time
        logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
        
        # Show processing results
        if parsers:
            successful_count = len([p for p in parsers if p.parse_status in ['Success', 'Fallback Success']])
            failed_count = len(parsers) - successful_count
            
            st.success(f"✅ Successfully processed {successful_count}/{len(uploaded_files)} files in {total_processing_time:.2f}s")
            
            if failed_count > 0:
                st.warning(f"⚠️ {failed_count} files failed to process. Check the Processing Summary for details.")
            
            create_dashboard(parsers)
            # Clear processing flag after successful processing
            st.session_state.processing_files = False
            # Reload saved portfolios to get fresh data
            st.session_state.saved_portfolios = st.session_state.db_manager.load_all_portfolios()
            # Force refresh to show new data
            st.rerun()
        else:
            st.error("❌ No valid PDF files could be parsed. Please check your files and try again.")
            logger.error("No files could be processed successfully")
            # Clear processing flag even if processing failed
            st.session_state.processing_files = False
    
    else:
        # Show database data if available, otherwise show welcome message
        if st.session_state.saved_portfolios:
            # Get selected portfolio from sidebar
            selected_portfolio = None
            if 'selected_portfolio' in st.session_state:
                selected_portfolio = st.session_state.selected_portfolio
            
            # Show dashboard with database data
            create_dashboard(selected_portfolio=selected_portfolio)
        else:
            # Show welcome message and instructions
            st.markdown("""
            # Welcome to CAMS Portfolio Analyzer 🚀
            
            This dashboard helps you analyze your mutual fund portfolio from CAMS account statements using advanced parsing technology with local database storage.
            
            ### Features:
            - 📊 **Portfolio Overview**: Get a complete view of your investments
            - 👥 **Investor Details**: PAN-wise investment breakdown and analysis
            - 🏦 **Fund Allocation**: See how your money is distributed across fund houses
            - 📈 **Performance Analytics**: Track returns and performance metrics
            - 📅 **Transaction History**: Review all your purchases and redemptions
            - 📋 **Detailed Holdings**: Comprehensive breakdown of all your funds
            - 🔍 **Advanced Parsing**: Uses casparser library for accurate data extraction
            - 💾 **Local Database**: Save and manage multiple portfolios locally
            - 🔄 **Portfolio Management**: Add, update, and delete portfolios easily
            
            ### Getting Started:
            1. **Upload New Files**: Use the sidebar to upload CAMS PDF files
            2. **Auto-save**: Enable auto-save to store portfolios in local database
            3. **View Saved**: Select from saved portfolios in the dropdown
            4. **Manage Data**: Use database management tools to organize your data
            
            ### Database Features:
            - **Local Storage**: All data stored in SQLite database (`portfolio_data.db`)
            - **Portfolio Management**: View, update, and delete saved portfolios
            - **Data Persistence**: Portfolios are automatically loaded on startup
            - **Export Options**: Download individual portfolios or entire database
            - **Statistics**: Track total portfolios, holdings, and portfolio value
            
            ### Supported Files:
            - CAMS Consolidated Account Statement PDFs
            - KFINTECH/KARVY PDFs
            - Multiple files can be uploaded and analyzed together
            - Password-protected PDFs are supported
            
            ### Technology:
            - Powered by [casparser](https://pypi.org/project/casparser/) library
            - SQLite database for local data storage
            - Supports CAMS, KFINTECH, and KARVY statement formats
            - Accurate parsing of holdings, transactions, and investor information
            """)

if __name__ == "__main__":
    main()
