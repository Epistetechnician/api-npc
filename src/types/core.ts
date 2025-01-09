import { ChainName, AssetType } from './providers';

export interface Chain {
  id: number;
  name: ChainName;
  is_active: boolean;
  created_at: Date;
  updated_at: Date;
}

export interface Asset {
  id: number;
  symbol: string;
  name: string;
  type: AssetType;
  chain_id: number;
  decimals?: number;
  contract_address?: string;
  created_at: Date;
  updated_at: Date;
}

export interface DataQuality {
  id: number;
  source_id: number;
  metric_type: string;
  timestamp: Date;
  completeness_score: number;
  accuracy_score: number;
  timeliness_score: number;
  consistency_score: number;
  validation_errors?: Record<string, any>;
  created_at: Date;
}

export interface AuditLog {
  id: number;
  table_name: string;
  record_id: number;
  action: string;
  old_data?: Record<string, any> | null;
  new_data?: Record<string, any>;
  user_id?: string;
  created_at: Date;
} 