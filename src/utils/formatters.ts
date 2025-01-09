export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2
  }).format(num);
};

export const formatPercentage = (num: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(num / 100);
};

export const formatCurrency = (num: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(num);
};

export const formatDate = (date: Date): string => {
  return date.toLocaleDateString()
}

export const convertToCSV = (data: any[]): string => {
  if (data.length === 0) return ''
  const headers = Object.keys(data[0])
  const rows = data.map(obj => headers.map(header => obj[header]))
  return [headers.join(','), ...rows.map(row => row.join(','))].join('\n')
} 