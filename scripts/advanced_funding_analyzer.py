import ccxt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from tabulate import tabulate
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
import time
from hyperliquid.info import Info
from hyperliquid.utils import constants
import aiohttp
import asyncio
from rich.table import Table
import requests
import traceback
import math
from supabase import create_client

# Load environment variables
load_dotenv()

# Setup logging and rich console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class AdvancedFundingAnalyzer:
    def __init__(self):
        """Initialize exchange connections with proper configuration"""
        # Initialize Binance with CCXT
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'defaultNetwork': 'BSC',
                'recvWindow': 60000,
            },
            'rateLimit': 100,
            'timeout': 30000
        })
        
        # Initialize Hyperliquid client
        self.hyperliquid = Info(
            base_url=constants.MAINNET_API_URL,
            request_timeout=30
        )

        # Initialize Supabase client properly
        try:
            self.supabase = create_client(
                supabase_url=os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
                supabase_key=os.getenv('NEXT_PUBLIC_SUPABASE_KEY'),
                options={
                    'headers': {
                        'Authorization': f'Bearer {os.getenv("NEXT_PUBLIC_SUPABASE_KEY")}',
                        'apikey': os.getenv('NEXT_PUBLIC_SUPABASE_KEY')
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
            self.supabase = None

    async def get_hyperliquid_rates(self) -> List[Dict]:
        """Fetch funding rates from Hyperliquid"""
        try:
            console.print("[cyan]Loading Hyperliquid markets...[/cyan]")
            
            # Get all markets info
            markets = await self.hyperliquid.get_all_mids()
            
            if not markets or 'allMids' not in markets:
                logger.error("No Hyperliquid markets data available")
                return []

            formatted_rates = []
            for market in markets['allMids']:
                try:
                    symbol = market['coin']
                    funding_rate = float(market.get('fundingRate', 0))
                    mark_price = float(market.get('markPrice', 0))
                    
                    formatted_rates.append({
                        'exchange': 'Hyperliquid',
                        'symbol': symbol,
                        'funding_rate': funding_rate,
                        'predicted_rate': funding_rate,  # Use current rate as prediction
                        'next_funding_time': datetime.now() + timedelta(hours=1),
                        'mark_price': mark_price,
                        'payment_interval': 1,  # Hyperliquid uses 1-hour intervals
                        'volume_24h': float(market.get('volume24h', 0)),
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    logger.warning(f"Error processing Hyperliquid market {market.get('coin', 'unknown')}: {e}")
                    continue

            if formatted_rates:
                console.print(f"[green]✓ Successfully fetched {len(formatted_rates)} Hyperliquid rates[/green]")
                
                # Display sample rates
                console.print("\nSample Hyperliquid rates:")
                for rate in formatted_rates[:3]:
                    console.print(f"Symbol: {rate['symbol']}, Rate: {rate['funding_rate']}, Predicted: {rate['predicted_rate']}")
                
            return formatted_rates

        except Exception as e:
            logger.error(f"Error fetching Hyperliquid rates: {e}")
            return []

    def get_binance_all_rates(self) -> List[Dict]:
        """Fetch funding rates from Binance with improved error handling"""
        try:
            console.print("[cyan]Loading Binance markets...[/cyan]")
            
            # Try using public API endpoints first
            endpoints = [
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                "https://api.binance.com/fapi/v1/premiumIndex"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(
                        endpoint,
                        timeout=15,
                        headers={
                            'User-Agent': 'Mozilla/5.0',
                            'Accept': 'application/json'
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        formatted_rates = []
                        
                        for item in data:
                            try:
                                if not isinstance(item, dict) or 'symbol' not in item:
                                    continue
                                    
                                if not item['symbol'].endswith('USDT'):
                                    continue
                                    
                                symbol = item['symbol'].replace('USDT', '')
                                funding_rate = float(item.get('lastFundingRate', 0)) * 100  # Convert to percentage
                                predicted_rate = float(item.get('predictedFundingRate', funding_rate)) * 100  # Convert to percentage
                                
                                formatted_rates.append({
                                    'exchange': 'Binance',
                                    'symbol': symbol,
                                    'funding_rate': funding_rate,
                                    'predicted_rate': predicted_rate,
                                    'next_funding_time': datetime.fromtimestamp(
                                        int(item.get('nextFundingTime', time.time() * 1000)) / 1000
                                    ),
                                    'mark_price': float(item.get('markPrice', 0)),
                                    'payment_interval': 8,  # Binance uses 8-hour intervals
                                    'volume_24h': 0,  # Not available in this endpoint
                                    'timestamp': datetime.now()
                                })
                            except Exception as e:
                                logger.warning(f"Error processing Binance item: {e}")
                                continue
                        
                        if formatted_rates:
                            console.print(f"[green]✓ Successfully fetched {len(formatted_rates)} Binance rates[/green]")
                            return formatted_rates
                            
                except Exception as e:
                    logger.warning(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            logger.error("All Binance endpoints failed")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Binance rates: {str(e)}")
            return []

    def analyze_funding_opportunities(self) -> pd.DataFrame:
        """Analyze funding opportunities across exchanges with improved processing"""
        try:
            # Get Hyperliquid rates first
            console.print("\n[cyan]Fetching Hyperliquid rates first...[/cyan]")
            hl_rates = asyncio.run(self.get_hyperliquid_rates())
            
            if not hl_rates:
                console.print("[red]❌ No Hyperliquid rates available[/red]")
            else:
                console.print(f"\n[green]✓ Successfully fetched {len(hl_rates)} Hyperliquid rates[/green]")
            
            # Get Binance rates
            binance_rates = self.get_binance_all_rates()
            
            # Combine and process rates
            all_rates = hl_rates + binance_rates
            if not all_rates:
                console.print("[red]❌ No funding rates data available[/red]")
                return pd.DataFrame()

            df = pd.DataFrame(all_rates)
            
            # Calculate annualized rates and other metrics
            df['annualized_rate'] = df.apply(
                lambda x: float(x['funding_rate']) * (365 * 24 / x['payment_interval']) * 100,
                axis=1
            )
            
            # Calculate rate differences and opportunity scores
            df['rate_diff'] = abs(df['predicted_rate'] - df['funding_rate'])
            df['opportunity_score'] = df.apply(self.calculate_opportunity_score, axis=1)
            
            # Add trading direction
            df['direction'] = df.apply(
                lambda x: 'long' if x['funding_rate'] < 0 else 'short',
                axis=1
            )

            return df.sort_values('opportunity_score', ascending=False)

        except Exception as e:
            logger.error(f"Error analyzing funding opportunities: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def calculate_opportunity_score(self, row: pd.Series) -> float:
        """Calculate opportunity score with improved weighting"""
        try:
            # Base score from rate difference
            score = abs(row['rate_diff']) * 100
            
            # Adjust based on payment interval
            interval_factor = 8.0 / row['payment_interval']
            score *= interval_factor
            
            # Adjust based on volume
            if row['volume_24h'] > 0:
                volume_factor = min(1.5, math.log10(row['volume_24h']) / 5)
                score *= volume_factor
            
            # Adjust based on mark price
            if row['mark_price'] > 0:
                price_factor = min(1.2, math.log10(row['mark_price']) / 4)
                score *= price_factor
                
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.0

    def analyze_arbitrage_opportunities(self, comparison_df: pd.DataFrame) -> List[Dict]:
        """Analyze and recommend arbitrage opportunities"""
        opportunities = []
        
        for _, row in comparison_df.iterrows():
            binance_rate = row['Binance Rate']
            hl_rate = row['HL Rate']
            spread = row['Spread']
            
            # Calculate annualized returns
            binance_annual = binance_rate * 365 * 100
            hl_annual = hl_rate * 365 * 100
            
            opportunity = {
                'symbol': row['Symbol'],
                'spread': abs(spread),
                'direction': 'Long Binance/Short HL' if binance_rate < hl_rate else 'Long HL/Short Binance',
                'expected_annual': abs(binance_annual - hl_annual),
                'binance_rate': binance_rate,
                'hl_rate': hl_rate,
                'binance_predicted': row['Binance Pred.'],
                'hl_predicted': row['HL Pred.']
            }
            
            opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x['spread'], reverse=True)

    def display_results(self, df: pd.DataFrame):
        """Enhanced display with predicted rates and better arbitrage recommendations"""
        console = Console()
        
        # Header with timestamp
        console.print("\n" + "="*80)
        console.print("🏦 Funding Rate Analysis Report", style="bold cyan")
        console.print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        console.print("="*80 + "\n")

        # Market Summary
        console.print("[yellow]📊 Market Summary[/yellow]")
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")
        
        summary_stats = {
            "Total Markets": len(df),
            "Average Funding Rate": f"{df['funding_rate'].mean():.6f}",
            "Median Funding Rate": f"{df['funding_rate'].median():.6f}",
            "Highest Rate": f"{df['funding_rate'].max():.6f}",
            "Lowest Rate": f"{df['funding_rate'].min():.6f}"
        }
        
        for metric, value in summary_stats.items():
            summary_table.add_row(metric, str(value))
        console.print(summary_table)
        
        # Existing comparison logic
        comparison_df = self._prepare_comparison_data(df)
        
        if not comparison_df.empty:
            console.print("\n[yellow]📊 Top Funding Rate Arbitrage Opportunities[/yellow]")
            
            # Enhanced arbitrage table
            arb_table = Table(
                show_header=True,
                header_style="bold magenta",
                title="Top 10 Arbitrage Opportunities",
                title_style="bold cyan"
            )
            
            # Add columns with proper formatting
            columns = [
                ("Symbol", "cyan"),
                ("Binance Rate", "green"),
                ("Binance Pred.", "blue"),
                ("Binance Ann.%", "yellow"),
                ("HL Rate", "green"),
                ("HL Pred.", "blue"),
                ("HL Ann.%", "yellow"),
                ("Spread", "red")
            ]
            
            for col_name, col_style in columns:
                arb_table.add_column(col_name, style=col_style)
            
            # Format and add rows
            for _, row in comparison_df.head(10).iterrows():
                arb_table.add_row(
                    row['Symbol'],
                    f"{row['Binance Rate']:.6f}",
                    f"{row['Binance Pred.']:.6f}",
                    f"{row['Binance Ann.%']:.2f}",
                    f"{row['HL Rate']:.6f}",
                    f"{row['HL Pred.']:.6f}",
                    f"{row['HL Ann.%']:.2f}",
                    f"{row['Spread']:.6f}"
                )
            
            console.print(arb_table)
            
            # Get top 10 symbols by spread
            top_symbols = comparison_df.head(10)['Symbol'].tolist()
            
            # Fetch predicted rates for top symbols
            predicted_rates = self.get_coinalyze_predicted_rates(top_symbols)
            
            # Enhanced arbitrage recommendations
            console.print("\n💡 [yellow]Advanced Arbitrage Recommendations:[/yellow]")
            opportunities = self.analyze_arbitrage_opportunities(comparison_df)
            
            for opp in opportunities[:5]:  # Top 5 opportunities
                predicted_rate = predicted_rates.get(opp['symbol'], None)
                
                console.print(
                    f"\n[cyan]• {opp['symbol']}[/cyan]",
                    style="bold"
                )
                console.print(
                    f"  Strategy: {opp['direction']}\n"
                    f"  Current Spread: {opp['spread']:.6f}\n"
                    f"  Expected Annual Return: {opp['expected_annual']:.2f}%\n"
                    f"  Binance Rate: {opp['binance_rate']:.6f} "
                    f"(Predicted: {opp['binance_predicted']:.6f})\n"
                    f"  Hyperliquid Rate: {opp['hl_rate']:.6f} "
                    f"(Predicted: {opp['hl_predicted']:.6f})"
                )
                
                if predicted_rate:
                    console.print(
                        f"  Coinalyze Predicted Rate: {predicted_rate:.6f}",
                        style="bright_cyan"
                    )
                
                # Add recommendation confidence
                spread_threshold = 0.0005  # 5 basis points
                if opp['spread'] > spread_threshold:
                    console.print("  Confidence: [green]High[/green] - Significant spread")
                else:
                    console.print("  Confidence: [yellow]Medium[/yellow] - Monitor spread")

    def _prepare_comparison_data(self, df: pd.DataFrame):
        """Helper method to prepare comparison data"""
        # Create comparison dataframes
        binance_df = df[df['exchange'] == 'Binance'].set_index('symbol')
        hl_df = df[df['exchange'] == 'Hyperliquid'].set_index('symbol')
        
        # Get common symbols
        common_symbols = set(binance_df.index) & set(hl_df.index)
        
        # Prepare comparison data
        comparison_data = []
        for symbol in common_symbols:
            comparison_data.append({
                'Symbol': symbol,
                'Binance Rate': binance_df.loc[symbol, 'funding_rate'],
                'Binance Pred.': binance_df.loc[symbol, 'predicted_rate'],
                'Binance Ann.%': binance_df.loc[symbol, 'annualized_rate'],
                'HL Rate': hl_df.loc[symbol, 'funding_rate'],
                'HL Pred.': hl_df.loc[symbol, 'predicted_rate'],
                'HL Ann.%': hl_df.loc[symbol, 'annualized_rate'],
                'Spread': binance_df.loc[symbol, 'funding_rate'] - hl_df.loc[symbol, 'funding_rate']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        if not comparison_df.empty:
            # Sort by absolute spread
            comparison_df['Abs_Spread'] = comparison_df['Spread'].abs()
            comparison_df = comparison_df.sort_values('Abs_Spread', ascending=False)
            comparison_df = comparison_df.drop('Abs_Spread', axis=1)
            
            return comparison_df
        else:
            return pd.DataFrame()

    def push_to_supabase(self, df: pd.DataFrame) -> bool:
        """Push analyzed data to Supabase"""
        try:
            if not self.supabase:
                logger.error("Supabase client not initialized")
                return False

            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Push market data
            try:
                self.supabase.table('funding_rate_snapshots').upsert(
                    records,
                    on_conflict='exchange,symbol,timestamp'
                ).execute()
                logger.info(f"Pushed {len(records)} market records")
            except Exception as e:
                logger.error(f"Error pushing market records: {e}")

            # Push statistics
            try:
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'total_markets': len(df),
                    'positive_funding': len(df[df['funding_rate'] > 0]),
                    'negative_funding': len(df[df['funding_rate'] < 0]),
                    'avg_funding_rate': df['funding_rate'].mean(),
                    'max_funding_rate': df['funding_rate'].max(),
                    'min_funding_rate': df['funding_rate'].min()
                }
                self.supabase.table('market_stats').upsert(
                    stats,
                    on_conflict='timestamp'
                ).execute()
                logger.info("Pushed statistics")
            except Exception as e:
                logger.error(f"Error pushing statistics: {e}")

            # Push top opportunities
            try:
                top_opps = df.nlargest(100, 'opportunity_score').to_dict('records')
                self.supabase.table('funding_top_opportunities').insert(top_opps).execute()
                logger.info(f"Pushed {len(top_opps)} top opportunities")
            except Exception as e:
                logger.error(f"Error pushing top opportunities: {e}")

            return True
            
        except Exception as e:
            logger.error(f"Error in push_to_supabase: {e}")
            return False

def main():
    try:
        console.print(Panel.fit(
            "🚀 Advanced Funding Rate Analysis",
            style="bold cyan"
        ))

        analyzer = AdvancedFundingAnalyzer()
        df = analyzer.analyze_funding_opportunities()
        
        if not df.empty:
            analyzer.display_results(df)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "data/funding_analysis"
            os.makedirs(output_dir, exist_ok=True)
            
            csv_file = f"{output_dir}/funding_analysis_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            console.print(f"\n💾 Results saved to: {csv_file}")
            
            console.print("\n💡 Trading Suggestions:")
            console.print("  • Consider long positions on assets with high negative funding rates")
            console.print("  • Consider short positions on assets with high positive funding rates")
            console.print("  • Look for funding rate arbitrage between exchanges")
            
    except Exception as e:
        console.print(f"[red]Error in main execution: {str(e)}[/red]")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 