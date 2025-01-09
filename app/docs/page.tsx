'use client';

import React, { useState } from 'react';
import { Book, Code, Database, FileText } from 'lucide-react';

interface DocState {
  loading: boolean;
  docs: any;
  activeCategory: string;
  activeType: string;
}

export default function DocsPage() {
  const [state, setState] = useState<DocState>({
    loading: true,
    docs: {},
    activeCategory: 'All',
    activeType: 'all'
  });
  
  const categories = ['All', 'DeFi', 'Analytics', 'GraphQL', 'NFTs', 'Infrastructure'];
  const types = [
    { id: 'all', label: 'All', icon: FileText },
    { id: 'api', label: 'APIs', icon: Database },
    { id: 'guide', label: 'Guides', icon: Book },
    { id: 'reference', label: 'Reference', icon: Code }
  ];

  const filteredDocs = React.useMemo(() => {
    let filtered = { ...state.docs };
    
 
    return filtered;
  }, [state.docs, state.activeCategory, state.activeType]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-black">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gray-900/50 border-b border-gray-700/50">
        <div className="max-w-7xl mx-auto py-16 px-4 sm:py-24 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
              Blockchain Data Documentation
            </h1>
            <p className="mt-4 text-xl text-gray-400">
              Comprehensive guides and references for blockchain data APIs
            </p>
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-purple-800/20 via-pink-800/20 to-red-800/20 animate-gradient-x" />
      </div>

      {/* Filters */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col sm:flex-row gap-4 justify-between">
          {/* Categories */}
          <div className="flex gap-2 overflow-x-auto pb-2">
            {categories.map(category => (
              <button
                key={category}
                onClick={() => setState(prev => ({ ...prev, activeCategory: category }))}
                className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap
                  ${state.activeCategory === category
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
              >
                {category}
              </button>
            ))}
          </div>

          {/* Types */}
          <div className="flex gap-2 overflow-x-auto pb-2">
            {types.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setState(prev => ({ ...prev, activeType: id }))}
                className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap flex items-center gap-2
                  ${state.activeType === id
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Documentation Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {Object.entries(filteredDocs).map(([name, doc]: [string, any]) => (
            <div key={name} className="bg-gray-800/50 rounded-lg border border-gray-700/50 overflow-hidden hover:border-purple-500/50 transition-colors">
              <div className="p-6">
                <div className="flex items-center gap-4 mb-4">
                  <div className="p-3 rounded-lg bg-purple-500/10 text-purple-400">
                    {doc.type === 'api' ? <Database className="h-6 w-6" /> :
                     doc.type === 'guide' ? <Book className="h-6 w-6" /> :
                     <Code className="h-6 w-6" />}
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-purple-400">{name}</h2>
                    <p className="text-gray-400 text-sm">{doc.category}</p>
                  </div>
                </div>
                
                <p className="text-gray-300 text-sm mb-4">{doc.metadata?.description || 'No description available'}</p>
                
                {doc.endpoints && (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-400 mb-2">Endpoints</h3>
                    <div className="space-y-2">
                      {doc.endpoints.slice(0, 3).map((endpoint: any) => (
                        <div key={endpoint.path} className="text-sm">
                          <code className="text-purple-400">{endpoint.method} {endpoint.path}</code>
                        </div>
                      ))}
                      {doc.endpoints.length > 3 && (
                        <p className="text-gray-500 text-sm">+ {doc.endpoints.length - 3} more endpoints</p>
                      )}
                    </div>
                  </div>
                )}

                {doc.examples && doc.examples.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-400 mb-2">Examples</h3>
                    <div className="space-y-2">
                      {doc.examples.slice(0, 2).map((example: any, index: number) => (
                        <div key={index} className="text-sm">
                          <p className="text-gray-400">{example.title}</p>
                          <code className="text-xs text-gray-500">{example.language}</code>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 