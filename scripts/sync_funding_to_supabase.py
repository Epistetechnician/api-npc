import os
import json
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import logging
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv
from tabulate import tabulate
import urllib.parse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseSync:
    def __init__(self):
        load_dotenv()
        
        # Get and validate Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")  # Changed from NEXT_PUBLIC_SUPABASE_URL
        supabase_key = os.getenv("SUPABASE_KEY")  # Changed from NEXT_PUBLIC_SUPABASE_KEY
        
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials in .env file")
        
        # Validate URL format
        try:
            parsed_url = urllib.parse.urlparse(supabase_url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError(f"Invalid Supabase URL format: {supabase_url}")
            
            # Initialize Supabase client
            self.supabase: Client = create_client(supabase_url, supabase_key)
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise

    def _test_connection(self):
        """Test the Supabase connection"""
        try:
            # Try a simple query to test connection
            self.supabase.table('funding_rate_snapshots').select("id").limit(1).execute()
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise

    def serialize_datetime(self, obj):
        """Helper method to serialize datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
        
    def calculate_notional_oi(self, market: Dict) -> float:
        """Calculate notional open interest if not present"""
        try:
            if 'notional_open_interest' in market:
                return float(market['notional_open_interest'])
            return float(market['open_interest']) * float(market['mark_price'])
        except (KeyError, TypeError, ValueError):
            return 0.0

    def process_funding_data(self, file_path: str) -> List[Dict]:
        """Process funding data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            funding_data = []
            for market in data.get('all_markets', []):
                # Calculate notional open interest
                notional_oi = float(market['mark_price']) * float(market['open_interest'])
                
                funding_data.append({
                    'timestamp': market['timestamp'],
                    'token': market['token'],
                    'current_funding_rate': market['current_funding_rate'],
                    'predicted_funding_rate': market['predicted_funding_rate'],
                    'mark_price': market['mark_price'],
                    'open_interest': market['open_interest'],
                    'notional_open_interest': notional_oi,
                    'volume_24h': market['volume_24h'],
                    'avg_24h_funding_rate': market.get('avg_24h_funding_rate', 0),
                    'exchange': 'hyperliquid'
                })
                
            return funding_data
        except Exception as e:
            logger.error(f"Error processing funding file {file_path}: {str(e)}")
            return []

    async def sync_funding_rates(self, data: Dict):
        """Sync funding rate data to Supabase tables"""
        try:
            if not isinstance(data, dict) or 'timestamp' not in data:
                raise ValueError("Invalid data format")

            timestamp = datetime.fromisoformat(data['timestamp'])
            
            # Validate and prepare market data
            if 'all_markets' not in data or not isinstance(data['all_markets'], list):
                raise ValueError("Missing or invalid market data")

            # Sort and rank markets by notional open interest
            sorted_markets = sorted(
                data['all_markets'],
                key=lambda x: self.calculate_notional_oi(x),
                reverse=True
            )
            
            # Prepare funding snapshots
            funding_snapshots = []
            for rank, market in enumerate(sorted_markets, 1):
                try:
                    snapshot = {
                        'timestamp': timestamp.isoformat(),
                        'token': market['token'],
                        'current_funding_rate': float(market['current_funding_rate']),
                        'predicted_funding_rate': float(market.get('predicted_funding_rate', 0)),
                        'annualized_funding': float(market['current_funding_rate']) * 365 * 100,
                        'mark_price': float(market['mark_price']),
                        'open_interest': float(market['open_interest']),
                        'notional_open_interest': self.calculate_notional_oi(market),
                        'open_interest_rank': rank,
                        'volume_24h': float(market['volume_24h']),
                        'avg_24h_funding_rate': float(market.get('avg_24h_funding_rate', 0)),
                        'exchange': 'hyperliquid',
                        'metadata': json.dumps({
                            'funding_difference': float(market.get('funding_difference', 0))
                        })
                    }
                    funding_snapshots.append(snapshot)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping market {market.get('token', 'unknown')}: {str(e)}")
                    continue

            # Insert funding snapshots
            if funding_snapshots:
                try:
                    # Use upsert to handle duplicates gracefully
                    result = self.supabase.table('funding_rate_snapshots').upsert(
                        funding_snapshots,
                        on_conflict='timestamp,token,exchange'
                    ).execute()
                    logger.info(f"✅ Successfully inserted {len(funding_snapshots)} funding rate snapshots")
                except Exception as e:
                    logger.error(f"Failed to insert funding snapshots: {str(e)}")
                    raise
            
            # Process opportunities
            if 'funding_opportunities' in data:
                opportunities = []
                for opp_type, opps in data['funding_opportunities'].items():
                    for opp in opps:
                        try:
                            opportunity = {
                                'timestamp': timestamp.isoformat(),
                                'token': opp['token'],
                                'opportunity_type': opp_type,
                                'current_rate': float(opp['current_funding_rate']),
                                'predicted_rate': float(opp.get('predicted_funding_rate', 0)),
                                'annualized_rate': float(opp.get('annualized_funding', 0)),
                                'rate_difference': float(opp.get('funding_difference', 0)),
                                'exchange': 'hyperliquid'
                            }
                            opportunities.append(opportunity)
                        except (KeyError, TypeError, ValueError) as e:
                            logger.warning(f"Skipping opportunity for {opp.get('token', 'unknown')}: {str(e)}")
                            continue

                if opportunities:
                    try:
                        result = self.supabase.table('funding_opportunities').insert(opportunities).execute()
                        logger.info(f"✅ Inserted {len(opportunities)} funding opportunities")
                    except Exception as e:
                        logger.error(f"Failed to insert opportunities: {str(e)}")
            
            # Update market stats
            if 'market_summary' in data:
                market_stats = {
                    'timestamp': timestamp.isoformat(),
                    'exchange': 'hyperliquid',
                    'total_markets': int(data['market_summary']['total_markets']),
                    'positive_funding_markets': int(data['market_summary']['positive_funding_markets']),
                    'negative_funding_markets': int(data['market_summary']['negative_funding_markets']),
                    'highest_annual_funding': float(data['market_summary']['highest_annual_funding']),
                    'lowest_annual_funding': float(data['market_summary']['lowest_annual_funding'])
                }
                
                try:
                    result = self.supabase.table('market_stats').upsert(
                        market_stats,
                        on_conflict='timestamp,exchange'
                    ).execute()
                    logger.info("✅ Updated market stats")
                except Exception as e:
                    logger.error(f"Failed to update market stats: {str(e)}")
            
            return funding_snapshots
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error syncing to Supabase: {str(e)}")
            raise

async def main():
    try:
        # Create data directory if it doesn't exist
        funding_dir = "data/funding_rates"
        os.makedirs(funding_dir, exist_ok=True)
        
        syncer = SupabaseSync()
        
        # Validate directory and files
        if not os.path.exists(funding_dir):
            raise FileNotFoundError(f"Directory not found: {funding_dir}")
            
        analysis_files = [f for f in os.listdir(funding_dir) if f.startswith('funding_analysis_')]
        
        if not analysis_files:
            raise FileNotFoundError("No funding analysis files found")
            
        latest_file = max(analysis_files)
        file_path = os.path.join(funding_dir, latest_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {latest_file}")
        
        # Process and sync data
        data = syncer.process_funding_data(file_path)
        if not data:
            raise ValueError("Failed to process funding data")
            
        funding_snapshots = await syncer.sync_funding_rates(data)
        if funding_snapshots:
            # Print summary
            df = pd.DataFrame(funding_snapshots)
            print("\n💰 Top 5 Markets by Open Interest:")
            print(tabulate(
                df.nlargest(5, 'notional_open_interest')[
                    ['token', 'open_interest', 'notional_open_interest', 'mark_price']
                ],
                headers=['Token', 'Open Interest', 'Notional OI (USD)', 'Price'],
                floatfmt=".2f",
                tablefmt="pretty"
            ))
            logger.info(f"Successfully synced funding data from {latest_file}")
            
    except FileNotFoundError as fe:
        logger.error(f"File error: {str(fe)}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 