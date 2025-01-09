export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat().format(num);
};

export const formatDate = (date: Date): string => {
  return date.toLocaleDateString();
};

export const convertToCSV = (data: any[]): string => {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]);
  const rows = data.map(obj => headers.map(header => obj[header]));
  
  return [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n');
}; 