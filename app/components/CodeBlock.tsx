'use client'

import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/cjs/styles/prism';

interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'typescript' }) => {
  return (
    <div className="rounded-lg overflow-hidden">
      {/* @ts-ignore - Known issue with react-syntax-highlighter types */}
      <SyntaxHighlighter
        language={language}
        style={tomorrow}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: 'rgba(0, 0, 0, 0.5)',
          borderRadius: '0.5rem',
        }}
        wrapLongLines={true}
        PreTag="div"
        useInlineStyles={true}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock; 