import { exec } from 'child_process'
import { promisify } from 'util'
import { NextResponse } from 'next/server'
import path from 'path'

const execAsync = promisify(exec)

export async function POST() {
  try {
    const scriptPath = path.join(process.cwd(), 'scripts/instant_process.py')
    const pythonPath = process.env.PYTHON_PATH || 'python'
    
    const { stdout, stderr } = await execAsync(`${pythonPath} ${scriptPath}`)
    
    if (stderr && !stderr.includes('DeprecationWarning')) {
      console.error('Script error:', stderr)
      return NextResponse.json({ error: stderr }, { status: 500 })
    }

    try {
      const result = JSON.parse(stdout)
      return NextResponse.json({
        success: true,
        message: 'Data refreshed successfully',
        records: result.records,
        timestamp: result.timestamp
      })
    } catch (e) {
      console.error('Failed to parse script output:', stdout)
      return NextResponse.json({ error: 'Invalid script output' }, { status: 500 })
    }

  } catch (error) {
    console.error('Refresh error:', error)
    return NextResponse.json({ 
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
} 