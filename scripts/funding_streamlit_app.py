import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from rich.console import Console
from advanced_funding_analyzer import AdvancedFundingAnalyzer
from st_supabase_connection import SupabaseConnection
import time
import logging
import yaml
from pathlib import Path
import sys
import logging.config
from supabase import create_client

# Configure logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'default',
            'level': 'INFO',
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    }
})

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize Supabase connection
def init_supabase():
    """Initialize Supabase connection using Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        supabase_url = st.secrets["connections"]["supabase"]["url"]
        supabase_key = st.secrets["connections"]["supabase"]["key"]
        
        # Create Supabase client
        client = create_client(supabase_url, supabase_key)
        
        # Test connection
        client.table('predicted_funding_rates').select('count', count='exact').limit(1).execute()
        
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

# Initialize the client at module level
supabase_client = init_supabase()

def get_predicted_rates():
    """Fetch predicted rates from Supabase with proper error handling"""
    try:
        if not supabase_client:
            logger.error("Supabase client not initialized")
            return pd.DataFrame()
        
        # Get most recent predictions for each asset
        response = supabase_client.table('predicted_funding_rates').select('*').execute()
        
        if not response.data:
            logger.warning("No predicted rates found in Supabase")
            return pd.DataFrame()
            
        df = pd.DataFrame(response.data)
        
        # Convert timestamps
        for col in ['next_funding_time', 'created_at', 'timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Get latest prediction for each asset
        latest_df = (df.sort_values('created_at', ascending=False)
                      .groupby('asset')
                      .first()
                      .reset_index())
        
        # Prepare result DataFrame with standardized columns
        result_df = pd.DataFrame({
            'symbol': latest_df['asset'].str.upper(),
            'predicted_rate': pd.to_numeric(latest_df['predicted_rate'], errors='coerce') * 100,
            'next_funding_time': latest_df['next_funding_time'],
            'exchange': latest_df['exchange'],
            'direction': latest_df['direction']
        })
        
        logger.info(f"Fetched {len(result_df)} predicted rates")
        return result_df
        
    except Exception as e:
        logger.error(f"Error fetching predicted rates: {e}")
        return pd.DataFrame()

def analyze_funding_data():
    """Analyze funding rates and merge with predictions"""
    try:
        # Initialize analyzer
        analyzer = AdvancedFundingAnalyzer()
        
        # Get current rates
        df = analyzer.analyze_funding_opportunities()
        if df.empty:
            logger.error("No funding rates data available")
            return pd.DataFrame()
            
        # Get predicted rates
        predicted_df = get_predicted_rates()
        
        # Merge current rates with predictions if available
        if not predicted_df.empty:
            # Standardize symbol names for joining
            df['symbol'] = df['symbol'].str.upper()
            predicted_df['symbol'] = predicted_df['symbol'].str.upper()
            
            # Merge on symbol and exchange
            df = df.merge(
                predicted_df[['symbol', 'exchange', 'predicted_rate']],
                on=['symbol', 'exchange'],
                how='left',
                suffixes=('', '_pred')
            )
            
            # Update predicted rate where available
            df['predicted_rate'] = df['predicted_rate_pred'].fillna(df['predicted_rate'])
            df = df.drop('predicted_rate_pred', axis=1)
            
            # Recalculate rate difference
            df['rate_diff'] = abs(df['predicted_rate'] - df['funding_rate'])
            
            logger.info(f"Merged {len(predicted_df)} predictions with {len(df)} current rates")
        else:
            logger.warning("No predicted rates available, using current rates")
            
        return df
        
    except Exception as e:
        logger.error(f"Error in analyze_funding_data: {e}")
        return pd.DataFrame()

def check_supabase_connection():
    """Check Supabase connection and data availability"""
    try:
        if not supabase_client:
            logger.error("Supabase client not initialized")
            return False
            
        # Test connection with a small query
        response = (supabase_client.table('predicted_funding_rates')
            .select('count', count='exact')
            .limit(1)
            .execute())
            
        return True if response else False
        
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        return False

def fetch_data():
    """Fetch and combine current and predicted rates"""
    try:
        # Get predicted rates first
        predicted_df = get_predicted_rates()
        logger.info(f"Predicted rates shape: {predicted_df.shape if not predicted_df.empty else 'Empty'}")
        
        # Get current rates
        analyzer = AdvancedFundingAnalyzer()
        df = analyzer.analyze_funding_opportunities()
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Standardize symbols for matching
            df['symbol'] = df['symbol'].str.upper()
            
            if not predicted_df.empty:
                # Ensure predicted_df has correct column names before merge
                predicted_df = predicted_df.rename(columns={
                    'asset': 'symbol',
                    'predicted_rate': 'predicted_rate_pred',
                    'annualized_rate': 'annualized_rate_pred'
                })
                
                # Log before merge
                logger.info(f"Predicted rates before merge: {predicted_df[['symbol', 'predicted_rate_pred']].head().to_dict('records')}")
                
                # Merge with predicted rates
                df = df.merge(
                    predicted_df[[
                        'symbol',
                        'predicted_rate_pred',
                        'direction',
                        'annualized_rate_pred'
                    ]],
                    on='symbol',
                    how='left'
                )
                
                # Handle the merged columns
                df['predicted_rate'] = df['predicted_rate_pred'].fillna(df['funding_rate'])
                df['annualized_rate'] = df['annualized_rate_pred'].fillna(
                    df['funding_rate'] * 365 * 24 / df['payment_interval']
                )
                
                # Clean up temporary columns
                df = df.drop(columns=[
                    'predicted_rate_pred', 
                    'annualized_rate_pred'
                ], errors='ignore')
                
                # Set default values
                df['direction'] = df['direction'].fillna('neutral')
                df['time_to_funding'] = 8.0  # Default to 8 hours
            else:
                df['predicted_rate'] = df['funding_rate']
                df['direction'] = 'neutral'
                df['time_to_funding'] = 8.0
                df['annualized_rate'] = df['funding_rate'] * 365 * 24 / df['payment_interval']
            
            # Calculate scores
            df['opportunity_score'] = df.apply(calculate_opportunity_score, axis=1)
            
            logger.info(f"Final shape: {df.shape}")
            logger.info(f"Final columns: {df.columns.tolist()}")
            
            # Ensure all required columns exist
            required_cols = [
                'symbol', 'exchange', 'funding_rate', 'predicted_rate',
                'time_to_funding', 'direction', 'annualized_rate',
                'opportunity_score', 'mark_price'
            ]
            
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            return df.sort_values('opportunity_score', ascending=False)
            
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        if 'df' in locals():
            logger.error(f"DataFrame columns: {df.columns.tolist() if not df.empty else 'Empty DataFrame'}")
        return pd.DataFrame()

def calculate_opportunity_score(row):
    """Enhanced opportunity score calculation"""
    try:
        current_rate = abs(float(row['funding_rate']))
        predicted_rate = abs(float(row.get('predicted_rate', current_rate)))
        time_to_funding = float(row.get('time_to_funding', 8))
        
        # Calculate rate difference
        rate_diff = abs(current_rate - predicted_rate)
        
        # Time factor (higher score for closer funding times)
        time_factor = max(0, min(1, (8 - time_to_funding) / 8))
        
        # Exchange factor (weight opportunities differently by exchange)
        exchange_factor = 1.1 if row['exchange'] == 'Binance' else 1.0
        
        # Direction factor (higher score if prediction matches current direction)
        direction_match = 1.2 if (
            (current_rate > 0 and predicted_rate > 0) or 
            (current_rate < 0 and predicted_rate < 0)
        ) else 1.0
        
        # Combine all factors
        base_score = (rate_diff * 0.7 + abs(current_rate) * 0.3)
        score = base_score * (1 + time_factor) * exchange_factor * direction_match
        
        # Annualize the score
        return score * (365 * 24 / row['payment_interval']) * 100
    except Exception as e:
        return 0

def calculate_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics from the funding data with multiple timeframes"""
    if df.empty:
        return {}
    
    try:
        # Calculate base rates
        avg_rate = df['funding_rate'].mean()
        max_rate = df['funding_rate'].max()
        min_rate = df['funding_rate'].min()
        
        # Calculate timeframe rates
        hourly = avg_rate * 100
        eight_hour = hourly * 8
        daily = hourly * 24
        
        return {
            'total_markets': len(df),
            'binance_markets': len(df[df['exchange'] == 'Binance']),
            'hl_markets': len(df[df['exchange'] == 'Hyperliquid']),
            'avg_funding_rate': avg_rate,
            'max_funding_rate': max_rate,
            'min_funding_rate': min_rate,
            'hourly_rate': hourly,
            'eight_hour_rate': eight_hour,
            'daily_rate': daily,
            'timestamp': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return {}

def create_visualizations(df: pd.DataFrame) -> dict:
    """Create visualization data for Streamlit"""
    try:
        return {
            'funding_distribution': px.box(
                df, x='exchange', y='funding_rate',
                title='Funding Rate Distribution by Exchange'
            ),
            'opportunity_scatter': px.scatter(
                df,
                x='funding_rate',
                y='predicted_rate',  # Changed from predicted_funding_rate
                color='exchange',
                hover_data=['symbol', 'annualized_rate', 'opportunity_score'],
                title='Current vs Predicted Funding Rates'
            ),
            'top_opportunities': df.nlargest(10, 'opportunity_score'),
            'funding_heatmap': px.density_heatmap(
                df,
                x='funding_rate',
                y='predicted_rate',  # Changed from predicted_funding_rate
                title='Funding Rate Density'
            )
        }
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return {}

def generate_hummingbot_config(df: pd.DataFrame) -> dict:
    """Generate simple Hummingbot config for top opportunities"""
    try:
        # Get top 10 opportunities
        top_opps = df.nlargest(10, 'opportunity_score')
        
        # Get best token with highest opportunity score
        best_token = top_opps.iloc[0]['symbol']
        
        # Calculate optimal parameters based on the selected token
        token_data = top_opps[top_opps['symbol'] == best_token].iloc[0]
        
        # Simple config structure
        config = {
            'markets': {},
            'candles_config': [],
            'controllers_config': [],
            'config_update_interval': 60,
            'script_file_name': 'v2_funding_rate_arb.py',
            'leverage': 5,
            'min_funding_rate_profitability': abs(float(token_data['funding_rate'])),
            'connectors': [
                'binance_perpetual',
                'hyperliquid_perpetual'
            ],
            'tokens': [
                best_token,
                best_token
            ],
            'position_size_quote': 100,
            'profitability_to_take_profit': max(0.01, abs(float(token_data['funding_rate'])) * 10),
            'funding_rate_diff_stop_loss': -abs(float(token_data['funding_rate'])),
            'trade_profitability_condition_to_enter': False
        }
        
        # Display the config in Streamlit
        st.code(f"""
markets: {{}}
candles_config: []
controllers_config: []
config_update_interval: 60
script_file_name: v2_funding_rate_arb.py
leverage: 5
min_funding_rate_profitability: {config['min_funding_rate_profitability']:.6f}
connectors:
- binance_perpetual
- hyperliquid_perpetual
tokens:
- '{best_token}'
- '{best_token}'
position_size_quote: 100
profitability_to_take_profit: {config['profitability_to_take_profit']:.6f}
funding_rate_diff_stop_loss: {config['funding_rate_diff_stop_loss']:.6f}
trade_profitability_condition_to_enter: false
        """, language='yaml')
        
        st.success(f"✅ Generated config for {best_token} with opportunity score: {token_data['opportunity_score']:.2f}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error generating config: {e}")
        return {}

def save_config_file(config: dict) -> str:
    """Save config to YAML file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"config_arb_{timestamp}.yml"
        
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        return filename
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return ""

def push_to_supabase(df: pd.DataFrame, stats: dict, viz_data: dict):
    """Push data to Supabase with better error handling"""
    try:
        if not supabase_client:
            logger.error("Supabase client not initialized")
            return False
            
        if df.empty:
            logger.warning("No data to push to Supabase")
            return False
            
        # Push funding rates
        funding_rates = df.apply(lambda x: {
            'symbol': x['symbol'],
            'exchange': x['exchange'],
            'funding_rate': float(x['funding_rate']),
            'predicted_rate': float(x.get('predicted_rate', 0)),
            'created_at': datetime.now().isoformat()
        }, axis=1).tolist()
        
        try:
            query = """
            INSERT INTO funding_rates (symbol, exchange, funding_rate, predicted_rate, created_at)
            VALUES :values
            """
            supabase_client.query(query, values=funding_rates).execute()
            logger.info(f"Pushed {len(funding_rates)} funding rates")
        except Exception as e:
            logger.error(f"Error pushing funding rates: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in push_to_supabase: {e}")
        return False

def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def main():
    try:
        # Check Supabase connection first
        if not supabase_client:
            st.error("Failed to connect to Supabase. Please check your credentials.")
            return
            
        st.title("Funding Rate Analysis")
        
        if st.button("🔄 Refresh Data"):
            with st.spinner("Fetching latest funding rates..."):
                df = analyze_funding_data()  # Use the new analysis function
                
                if df.empty:
                    st.error("Failed to fetch funding rates data")
                    return
                    
                st.session_state.df = df
                st.session_state.last_update = datetime.now()
                st.session_state.stats = calculate_stats(df)
                
                # Create visualizations
                viz_data = create_visualizations(df)
                st.session_state.viz_data = viz_data
                
                st.success("Data refreshed successfully!")

        # Display data if available
        if 'df' in st.session_state and not st.session_state.df.empty:
            try:
                # Display metrics
                stats = st.session_state.stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Markets",
                        f"{stats['total_markets']}",
                        f"{stats['binance_markets']} Binance / {stats['hl_markets']} HL"
                    )
                with col2:
                    st.metric(
                        "1H Rate",
                        f"{stats['hourly_rate']:.4f}%",
                        f"{stats['hourly_rate']*365*24:.1f}% APR"
                    )
                with col3:
                    st.metric(
                        "8H Rate",
                        f"{stats['eight_hour_rate']:.4f}%",
                        f"{stats['eight_hour_rate']*365/8:.1f}% APR"
                    )
                with col4:
                    st.metric(
                        "24H Rate",
                        f"{stats['daily_rate']:.4f}%",
                        f"{stats['daily_rate']*365/24:.1f}% APR"
                    )

                # Create tabs
                tab1, tab2, tab3 = st.tabs([
                    "🎯 Top Opportunities",
                    "📊 Market Analysis",
                    "🔍 Detailed View"
                ])

                # Get visualization data
                viz_data = create_visualizations(st.session_state.df)

                # Display tabs content
                with tab1:
                    if 'opportunity_scatter' in viz_data:
                        st.plotly_chart(viz_data['opportunity_scatter'], use_container_width=True)
                    if 'top_opportunities' in viz_data:
                        display_top_opportunities(viz_data['top_opportunities'])

                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'funding_distribution' in viz_data:
                            st.plotly_chart(viz_data['funding_distribution'], use_container_width=True)
                    with col2:
                        if 'funding_heatmap' in viz_data:
                            st.plotly_chart(viz_data['funding_heatmap'], use_container_width=True)

                with tab3:
                    display_detailed_view(st.session_state.df)

            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
                logger.error(f"Display error: {str(e)}", exc_info=True)
                if st.button("Retry Display"):
                    st.rerun()
        else:
            st.info("Waiting for data... Click Refresh Data to start.")

        # Simplified config generation
        if st.button("Generate Hummingbot Config"):
            if 'df' in st.session_state and not st.session_state.df.empty:
                with st.spinner("Generating Hummingbot config..."):
                    generate_hummingbot_config(st.session_state.df)
            else:
                st.warning("Please fetch funding rates data first")

        # Add health check endpoint
        if 'health_check' in st.experimental_get_query_params():
            st.json(health_check())
            st.stop()

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {str(e)}")
        if st.button("Retry"):
            st.experimental_rerun()

def display_top_opportunities(top_opps):
    """Display top opportunities table with enhanced formatting"""
    if not top_opps.empty:
        display_df = top_opps.copy()
        
        # Ensure all required columns exist
        display_df['predicted_rate'] = display_df['predicted_rate'].fillna(display_df['funding_rate'])
        display_df['rate_diff'] = abs(display_df['predicted_rate'] - display_df['funding_rate'])
        display_df['next_funding'] = display_df['time_to_funding'].apply(
            lambda x: f"{x:.1f}h" if pd.notnull(x) else "8h"
        )
        display_df['suggested_position'] = display_df.apply(
            lambda x: "Long" if x['funding_rate'] < 0 else "Short", 
            axis=1
        )
        
        # Use custom styling without matplotlib dependency
        def style_negative(v):
            return 'color: red' if v < 0 else 'color: green'
            
        def style_opportunity(v):
            return f'background-color: rgba(0, 255, 0, {min(abs(v)/10, 0.5)})'
        
        # Display with enhanced formatting
        styled_df = display_df[[
            'symbol',
            'exchange',
            'funding_rate',
            'predicted_rate',
            'rate_diff',
            'next_funding',
            'direction',
            'suggested_position',
            'annualized_rate',
            'opportunity_score'
        ]].style.format({
            'funding_rate': '{:.4f}%',  # Show as percentage with 4 decimal places
            'predicted_rate': '{:.4f}%',
            'rate_diff': '{:.4f}%',
            'annualized_rate': '{:.2f}%',
            'opportunity_score': '{:.4f}'
        }).applymap(style_negative, subset=['funding_rate', 'rate_diff']
        ).applymap(style_opportunity, subset=['opportunity_score'])
        
        st.dataframe(styled_df)

def display_detailed_view(df):
    """Display enhanced detailed view of all data"""
    if not df.empty:
        display_df = df.copy()
        
        # Add formatted columns
        display_df['rate_diff'] = abs(display_df['predicted_rate'] - display_df['funding_rate'])
        display_df['next_funding'] = display_df['time_to_funding'].apply(
            lambda x: f"{x:.1f}h" if pd.notnull(x) else "8h"
        )
        display_df['suggested_position'] = display_df.apply(
            lambda x: "Long" if x['funding_rate'] < 0 else "Short", 
            axis=1
        )
        
        # Sort by opportunity score
        display_df = display_df.sort_values('opportunity_score', ascending=False)
        
        # Display with formatting
        st.dataframe(
            display_df[[
                'symbol',
                'exchange',
                'funding_rate',
                'predicted_rate',
                'rate_diff',
                'next_funding',
                'direction',
                'suggested_position',
                'annualized_rate',
                'opportunity_score',
                'mark_price'
            ]].style.format({
                'funding_rate': '{:.4f}%',
                'predicted_rate': '{:.4f}%',
                'rate_diff': '{:.4f}%',
                'annualized_rate': '{:.2f}%',
                'opportunity_score': '{:.4f}',
                'mark_price': '${:,.2f}'
            }).background_gradient(
                subset=['opportunity_score'],
                cmap='RdYlGn'
            ).background_gradient(
                subset=['rate_diff'],
                cmap='YlOrRd'
            )
        )

if __name__ == "__main__":
    main() 