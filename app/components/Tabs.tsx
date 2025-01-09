'use client'

import React from 'react'

interface TabsProps {
  children: React.ReactNode;
}

export const Tabs: React.FC<TabsProps> = ({ children }) => {
  return (
    <div className="w-full">
      {children}
    </div>
  );
};

export const TabList: React.FC<{children: React.ReactNode}> = ({ children }) => {
  return <div className="flex border-b border-gray-700">{children}</div>
}

interface TabProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

export const Tab: React.FC<TabProps> = ({ active, onClick, children }) => {
  return (
    <button
      className={`tab ${active ? 'active' : ''}`}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

interface TabPanelProps {
  id: string
  children: React.ReactNode
  activeTab?: string
}

export const TabPanel: React.FC<TabPanelProps> = ({ id, children, activeTab }) => {
  if (activeTab !== id) {
    return null
  }
  
  return <div className="py-4">{children}</div>
} 