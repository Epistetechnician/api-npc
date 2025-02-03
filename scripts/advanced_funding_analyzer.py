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

# Load environment variables
load_dotenv()

# Setup logging and rich console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class AdvancedFundingAnalyzer:
    def __init__(self):
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
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
            self.binance.load_markets()
            
            formatted_rates = []
            markets = [s for s in self.binance.symbols if s.endswith(':USDT')]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]Fetching Binance rates...", total=len(markets))
                
                for symbol in markets:
                    try:
                        funding_rate = self.binance.fetch_funding_rate(symbol)
                        
                        base = symbol.split('/')[0].replace(':USDT', '')
                        formatted_rates.append({
                            'exchange': 'Binance',
                            'symbol': base,
                            'funding_rate': float(funding_rate['fundingRate']),
                            'predicted_rate': float(funding_rate.get('predictedFundingRate', 0)),
                            'next_funding_time': datetime.fromtimestamp(funding_rate['fundingTimestamp'] / 1000),
                            'mark_price': float(funding_rate.get('markPrice', 0)),
                            'payment_interval': 8
                        })
                        progress.update(task, advance=1)
                        time.sleep(self.binance.rateLimit / 1000)  # Respect rate limits
                        
                    except Exception as e:
                        logger.warning(f"Error processing Binance rate for {symbol}: {e}")
                        progress.update(task, advance=1)
                        continue
            
            return formatted_rates
            
        except Exception as e:
            logger.error(f"Error fetching Binance rates: {e}")
            return []

    def get_hyperliquid_all_rates(self) -> List[Dict]:
        """Fetch funding rates from Hyperliquid using CCXT"""
        try:
            console.print("[cyan]Loading Hyperliquid markets...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("[cyan]Fetching Hyperliquid rates...", total=1)
                
                # Use CCXT's fetchFundingRates method
                funding_rates = self.hyperliquid.fetch_funding_rates()
                progress.update(task, advance=1)
                
                formatted_rates = []
                for symbol, data in funding_rates.items():
                    try:
                        # Extract base symbol (remove USDT)
                        base = symbol.split('/')[0]
                        
                        formatted_rates.append({
                            'exchange': 'Hyperliquid',
                            'symbol': base,
                            'funding_rate': float(data['fundingRate']),
                            'predicted_rate': float(data.get('predictedRate', 0)),
                            'next_funding_time': datetime.fromtimestamp(data['fundingTimestamp'] / 1000) if data.get('fundingTimestamp') else datetime.now() + timedelta(hours=1),
                            'mark_price': float(data.get('markPrice', 0)),
                            'payment_interval': 1
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing Hyperliquid rate for {symbol}: {e}")
                        continue
                
                return formatted_rates

        except Exception as e:
            logger.error(f"Error fetching Hyperliquid rates: {str(e)}")
            # Enable verbose mode for debugging
            self.hyperliquid.verbose = True
            return []

    def analyze_funding_opportunities(self) -> pd.DataFrame:
        """Analyze funding opportunities across exchanges"""
        try:
            # Get Hyperliquid rates first with debugging
            console.print("\n[cyan]Fetching Hyperliquid rates first...[/cyan]")
            hl_rates = self.get_hyperliquid_all_rates()
            
            if not hl_rates:
                console.print("[red]❌ No Hyperliquid rates available[/red]")
                return pd.DataFrame()
            
            # Debug output for Hyperliquid rates
            console.print(f"\n[green]✓ Successfully fetched {len(hl_rates)} Hyperliquid rates[/green]")
            console.print("\nSample Hyperliquid rates:")
            for rate in hl_rates[:3]:
                console.print(f"Symbol: {rate['symbol']}, Rate: {rate['funding_rate']}, Predicted: {rate['predicted_rate']}")
            
            # Get Binance rates
            console.print("\n[cyan]Fetching Binance rates...[/cyan]")
            binance_rates = self.get_binance_all_rates()

            # Combine and process rates
            all_rates = hl_rates + binance_rates
            if not all_rates:
                console.print("[red]❌ No funding rates data available[/red]")
                return pd.DataFrame()

            df = pd.DataFrame(all_rates)
            df['annualized_rate'] = df.apply(
                lambda x: float(x['funding_rate']) * (365 * 24 / x['payment_interval']) * 100,
                axis=1
            )

            return df

        except Exception as e:
            logger.error(f"Error analyzing funding opportunities: {e}")
            return pd.DataFrame()

    def display_results(self, df: pd.DataFrame):
        """Display results in a clear, organized format with side-by-side comparison"""
        console.print("\n" + "="*120)
        console.print("[cyan]🏦 Funding Rate Comparison - Binance vs Hyperliquid[/cyan]")
        console.print("="*120)
        
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
            
            console.print("\n[yellow]📊 Top Funding Rate Arbitrage Opportunities[/yellow]")
            console.print(tabulate(
                comparison_df.head(10),
                headers=[
                    'Symbol',
                    'Binance Rate',
                    'Binance Pred.',
                    'Binance Ann.%',
                    'HL Rate',
                    'HL Pred.',
                    'HL Ann.%',
                    'Spread'
                ],
                floatfmt=".6f",
                tablefmt="pretty"
            ))
        
        # Display individual exchange top rates
        for exchange in ['Binance', 'Hyperliquid']:
            exchange_df = df[df['exchange'] == exchange]
            if exchange_df.empty:
                continue
            
            console.print(f"\n{'='*120}")
            console.print(f"[cyan]🏦 {exchange} Top Funding Rates[/cyan]")
            console.print(f"{'='*120}")
            
            # Top positive rates
            console.print("\n[green]📈 Highest Positive Funding Rates:[/green]")
            positive_df = exchange_df.nlargest(10, 'funding_rate')[
                ['symbol', 'funding_rate', 'predicted_rate', 'annualized_rate', 'mark_price']
            ]
            console.print(tabulate(
                positive_df,
                headers=[
                    f'{exchange} Symbol',
                    'Current Rate',
                    'Predicted Rate',
                    'Annual %',
                    'Mark Price'
                ],
                floatfmt=".6f",
                tablefmt="pretty"
            ))
            
            # Top negative rates
            console.print("\n[red]📉 Lowest Negative Funding Rates:[/red]")
            negative_df = exchange_df.nsmallest(10, 'funding_rate')[
                ['symbol', 'funding_rate', 'predicted_rate', 'annualized_rate', 'mark_price']
            ]
            console.print(tabulate(
                negative_df,
                headers=[
                    f'{exchange} Symbol',
                    'Current Rate',
                    'Predicted Rate',
                    'Annual %',
                    'Mark Price'
                ],
                floatfmt=".6f",
                tablefmt="pretty"
            ))

        # Add arbitrage suggestions
        if not comparison_df.empty:
            console.print("\n💡 [yellow]Arbitrage Opportunities:[/yellow]")
            for _, row in comparison_df.head(5).iterrows():
                if row['Spread'] > 0:
                    console.print(f"  • {row['Symbol']}: Long Hyperliquid ({row['HL Rate']:.6f}) / Short Binance ({row['Binance Rate']:.6f}) - Spread: {row['Spread']:.6f}")
                else:
                    console.print(f"  • {row['Symbol']}: Long Binance ({row['Binance Rate']:.6f}) / Short Hyperliquid ({row['HL Rate']:.6f}) - Spread: {abs(row['Spread']):.6f}")

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