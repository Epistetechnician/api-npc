from supabase import create_client
import os
from dotenv import load_dotenv
import logging
from postgrest.exceptions import APIError

logger = logging.getLogger(__name__)

def setup_supabase_tables():
    """Create required Supabase tables if they don't exist"""
    load_dotenv()
    
    try:
        supabase = create_client(
            os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
            os.getenv("NEXT_PUBLIC_SUPABASE_KEY")
        )
        
        if not os.getenv("NEXT_PUBLIC_SUPABASE_URL") or not os.getenv("NEXT_PUBLIC_SUPABASE_KEY"):
            logger.error("Missing Supabase credentials in environment variables")
            return False

        # Create funding_rates table
        create_funding_rates = """
        create table if not exists public.funding_rates (
            id bigint primary key generated always as identity,
            symbol text not null,
            exchange text not null,
            funding_rate numeric not null,
            predicted_rate numeric,
            mark_price numeric,
            next_funding_time timestamptz,
            payment_interval integer,
            annualized_rate numeric,
            timestamp timestamptz default now(),
            created_at timestamptz default now()
        );
        """
        
        # Create funding_arbitrage_opportunities table
        create_arbitrage_opps = """
        create table if not exists public.funding_arbitrage_opportunities (
            id bigint primary key generated always as identity,
            coin text not null,
            exchange text not null,
            funding_rate numeric not null,
            predicted_rate numeric,
            mark_price numeric,
            volume_24h numeric,
            priority_score numeric,
            market_size numeric,
            timestamp timestamptz default now(),
            created_at timestamptz default now()
        );
        """
        
        # Try direct SQL execution first
        try:
            # Test connection
            supabase.table("funding_rates").select("count").limit(1).execute()
            logger.info("Connected to Supabase successfully")
            
            # Tables already exist
            return True
            
        except APIError as e:
            if 'relation "public.' in str(e):
                try:
                    # Try creating tables using direct SQL
                    supabase.rest.sql().execute(create_funding_rates)
                    supabase.rest.sql().execute(create_arbitrage_opps)
                    logger.info("Created tables successfully")
                    return True
                except Exception as sql_error:
                    logger.error(f"Failed to create tables: {sql_error}")
                    return False
            else:
                logger.error(f"Unexpected API error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"Error setting up Supabase tables: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if setup_supabase_tables():
        print("✅ Supabase tables setup complete")
    else:
        print("❌ Failed to setup Supabase tables") 