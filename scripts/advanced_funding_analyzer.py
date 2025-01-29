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
                'fetchFundingRateHistory': {
                    'limit': 1000,
                }
            },
            'rateLimit': 100,
            'timeout': 30000,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        })
        
        # Use CCXT for Hyperliquid
        self.hyperliquid = ccxt.hyperliquid({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # for perpetual futures
                'adjustForTimeDifference': True
            }
        })

    def get_binance_all_rates(self) -> List[Dict]:
        """Fetch both current and predicted funding rates from Binance"""
        try:
            console.print("[cyan]Loading Binance markets...[/cyan]")
            
            # Try using public API endpoints
            endpoints = [
                "https://api.allorigins.win/raw?url=https://fapi.binance.com/fapi/v1/premiumIndex",
                "https://api.codetabs.com/v1/proxy?quest=https://fapi.binance.com/fapi/v1/premiumIndex",
                "https://cors-anywhere.herokuapp.com/https://fapi.binance.com/fapi/v1/premiumIndex"
            ]
            
            for proxy_url in endpoints:
                try:
                    response = requests.get(
                        proxy_url,
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
                                predicted_rate = float(item.get('predictedFundingRate', 0)) * 100  # Convert to percentage
                                
                                # Calculate annualized rates (8-hour intervals)
                                annualized_rate = funding_rate * (365 * 3)  # 3 funding periods per day
                                
                                formatted_rates.append({
                                    'exchange': 'Binance',
                                    'symbol': symbol,
                                    'funding_rate': funding_rate,
                                    'predicted_rate': predicted_rate,
                                    'next_funding_time': datetime.fromtimestamp(
                                        int(item.get('nextFundingTime', time.time() * 1000)) / 1000
                                    ),
                                    'mark_price': float(item.get('markPrice', 0)),
                                    'payment_interval': 8,
                                    'volume_24h': 0,
                                    'timestamp': datetime.now(),
                                    'annualized_rate': annualized_rate,
                                    'rate_diff': abs(predicted_rate - funding_rate)
                                })
                            except Exception as e:
                                logger.warning(f"Error processing item {item}: {e}")
                                continue
                        
                        if formatted_rates:
                            console.print(f"[green]✓ Successfully fetched {len(formatted_rates)} Binance rates[/green]")
                            return formatted_rates
                            
                except Exception as e:
                    logger.warning(f"Proxy endpoint {proxy_url} failed: {e}")
                    continue
            
            # If all proxies fail, try direct API call with VPN headers
            try:
                response = requests.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    timeout=15,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/json',
                        'X-Forwarded-For': '8.8.8.8',  # Google DNS IP
                        'Origin': 'https://www.binance.com'
                    }
                )
                
                if response.status_code == 200:
                    # Process response same as above...
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
                            predicted_rate = float(item.get('predictedFundingRate', 0)) * 100  # Convert to percentage
                            
                            # Calculate annualized rates (8-hour intervals)
                            annualized_rate = funding_rate * (365 * 3)  # 3 funding periods per day
                            
                            formatted_rates.append({
                                'exchange': 'Binance',
                                'symbol': symbol,
                                'funding_rate': funding_rate,
                                'predicted_rate': predicted_rate,
                                'next_funding_time': datetime.fromtimestamp(
                                    int(item.get('nextFundingTime', time.time() * 1000)) / 1000
                                ),
                                'mark_price': float(item.get('markPrice', 0)),
                                'payment_interval': 8,
                                'volume_24h': 0,
                                'timestamp': datetime.now(),
                                'annualized_rate': annualized_rate,
                                'rate_diff': abs(predicted_rate - funding_rate)
                            })
                        except Exception as e:
                            logger.warning(f"Error processing item {item}: {e}")
                            continue
                    
                    if formatted_rates:
                        console.print(f"[green]✓ Successfully fetched {len(formatted_rates)} Binance rates directly[/green]")
                        return formatted_rates
                        
            except Exception as e:
                logger.error(f"Direct API call failed: {e}")
            
            logger.error("All attempts to fetch Binance rates failed")
            return []
            
        except Exception as e:
            logger.error(f"Error in get_binance_all_rates: {e}")
            return []

    def get_hyperliquid_all_rates(self) -> List[Dict]:
        """Fetch funding rates from Hyperliquid using CCXT"""
        try:
            formatted_rates = []
            
            try:
                markets = self.hyperliquid.load_markets()
                funding_rates = self.hyperliquid.fetch_funding_rates()
                
                for symbol, data in funding_rates.items():
                    try:
                        base = symbol.split('/')[0]
                        funding_rate = float(data['fundingRate']) * 100  # Convert to percentage
                        predicted_rate = float(data.get('predictedRate', 0)) * 100  # Convert to percentage
                        
                        # Calculate annualized rates (1-hour intervals)
                        annualized_rate = funding_rate * (365 * 24)  # 24 funding periods per day
                        
                        formatted_rates.append({
                            'exchange': 'Hyperliquid',
                            'symbol': base,
                            'funding_rate': funding_rate,
                            'predicted_rate': predicted_rate,
                            'next_funding_time': datetime.fromtimestamp(data.get('fundingTimestamp', time.time()) / 1000),
                            'mark_price': float(data.get('markPrice', 0)),
                            'payment_interval': 1,
                            'annualized_rate': annualized_rate,
                            'rate_diff': abs(predicted_rate - funding_rate)
                        })
                    except Exception as e:
                        logger.warning(f"Error processing Hyperliquid rate for {symbol}: {e}")
                        continue
                
                return formatted_rates

            except Exception as e:
                logger.error(f"Error fetching Hyperliquid rates: {str(e)}")
                return []

        except Exception as e:
            logger.error(f"Error in Hyperliquid rate fetch: {str(e)}")
            return []

    async def _fetch_single_coinalyze_rate(self, session, symbol: str) -> Optional[float]:
        """Fetch predicted rate for a single symbol"""
        try:
            # Format symbol correctly for Coinalyze API
            formatted_symbol = f"{symbol}USDT_PERP.A"  # Adding _PERP.A suffix as required
            url = f"https://api.coinalyze.net/v1/predicted-funding-rate"
            params = {'symbols': formatted_symbol}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract rate from response
                    if data and len(data) > 0:
                        return float(data[0].get('predictedRate', 0))
                else:
                    logger.warning(f"Could not fetch Coinalyze rate for {symbol}: Status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching Coinalyze rate for {symbol}: {e}")
            return None

    async def _fetch_coinalyze_rates_async(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch all predicted rates asynchronously"""
        api_key = os.getenv('COINANALYZE_API_KEY')
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
        # Process symbols in batches of 20 as per API limit
        batch_size = 20
        predicted_rates = {}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                # Format symbols for batch request
                formatted_symbols = [f"{s}USDT_PERP.A" for s in batch_symbols]
                
                url = "https://api.coinalyze.net/v1/predicted-funding-rate"
                params = {'symbols': ','.join(formatted_symbols)}
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Process batch response
                            for item in data:
                                symbol = item['symbol'].replace('USDT_PERP.A', '')
                                predicted_rates[symbol] = float(item.get('predictedRate', 0))
                        else:
                            logger.warning(f"Batch request failed: Status {response.status}")
                except Exception as e:
                    logger.error(f"Error in batch request: {e}")
                
                # Respect rate limits
                await asyncio.sleep(0.5)  # 500ms delay between batches
        
        return predicted_rates

    def get_coinalyze_predicted_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Synchronous wrapper for async predicted rates fetch"""
        try:
            return asyncio.run(self._fetch_coinalyze_rates_async(symbols))
        except Exception as e:
            logger.error(f"Error in Coinalyze API call: {e}")
            return {}

    def analyze_funding_opportunities(self) -> pd.DataFrame:
        """Analyze funding rates and calculate opportunities"""
        try:
            # Fetch rates from both exchanges
            binance_rates = self.get_binance_all_rates()
            hyperliquid_rates = self.get_hyperliquid_all_rates()
            
            # Combine rates
            all_rates = binance_rates + hyperliquid_rates
            
            if not all_rates:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(all_rates)
            
            # Calculate time to funding
            df['time_to_funding'] = (df['next_funding_time'] - datetime.now()).dt.total_seconds() / 3600
            
            # Calculate opportunity score
            df['opportunity_score'] = df.apply(lambda x: self._calculate_opportunity_score(
                funding_rate=x['funding_rate'],
                predicted_rate=x['predicted_rate'],
                time_to_funding=x['time_to_funding'],
                payment_interval=x['payment_interval']
            ), axis=1)
            
            # Add trading direction
            df['direction'] = df['funding_rate'].apply(lambda x: 'Long' if x < 0 else 'Short')
            
            return df
            
        except Exception as e:
            logger.error(f"Error in analyze_funding_opportunities: {e}")
            return pd.DataFrame()

    def _calculate_opportunity_score(self, funding_rate: float, predicted_rate: float, 
                                   time_to_funding: float, payment_interval: int) -> float:
        """Calculate opportunity score with proper rate handling"""
        try:
            # Normalize rates to hourly basis
            hourly_rate = funding_rate / payment_interval
            hourly_predicted = predicted_rate / payment_interval
            
            # Calculate score components
            rate_magnitude = abs(hourly_rate)
            rate_consistency = 1 - (abs(hourly_rate - hourly_predicted) / (rate_magnitude + 1e-10))
            time_factor = 1 - (time_to_funding / payment_interval)
            
            # Combine components
            score = rate_magnitude * rate_consistency * time_factor
            return round(score, 6)
            
        except Exception as e:
            logger.warning(f"Error calculating opportunity score: {e}")
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

if __name__ == "__main__":
    main() 