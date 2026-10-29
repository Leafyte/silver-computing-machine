import { spawn } from 'child_process';
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const outDir = path.resolve('../docs/screenshots');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

async function run() {
  console.log('Starting preview server on port 4173…');
  const server = spawn('npx', ['vite', 'preview', '--port', '4173', '--host', '127.0.0.1'], {
    shell: true,
    stdio: 'inherit'
  });

  // Wait 3 seconds for preview server
  await new Promise((r) => setTimeout(r, 3000));

  console.log('Launching browser…');
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();

  console.log('Navigating to http://127.0.0.1:4173…');
  await page.goto('http://127.0.0.1:4173', { waitUntil: 'domcontentloaded', timeout: 15000 });
  await page.waitForTimeout(2500);

  // 1. Command Center 1920x1080
  console.log('Capturing 01_command_center_1920.png');
  await page.screenshot({ path: path.join(outDir, '01_command_center_1920.png'), fullPage: true });

  // 2. Command Center 1440x900
  console.log('Capturing 02_command_center_1440.png');
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '02_command_center_1440.png') });

  // Reset viewport
  await page.setViewportSize({ width: 1920, height: 1080 });

  // 3. Switch to Fleet Operations
  console.log('Capturing 03_fleet_operations.png');
  await page.click('button[title="Fleet Operations"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '03_fleet_operations.png') });

  // 4. Switch to Route Planning
  console.log('Capturing 04_dispatch_planning.png');
  await page.click('button[title="Route Planning"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '04_dispatch_planning.png') });

  // 5. Switch to SLA Risk
  console.log('Capturing 05_sla_risk.png');
  await page.click('button[title="SLA & Exposure"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '05_sla_risk.png') });

  // 6. Switch to QML Intelligence
  console.log('Capturing 06_qml_evidence.png');
  await page.click('button[title="QML Intelligence"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '06_qml_evidence.png') });

  // 7. Switch to Audit Events
  console.log('Capturing 07_audit_events.png');
  await page.click('button[title="Audit Events"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '07_audit_events.png') });

  // 8. Switch to Solver Jobs
  console.log('Capturing 08_solver_jobs.png');
  await page.click('button[title="Solver Jobs"]');
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(outDir, '08_solver_jobs.png') });

  // 9. Switch to Solomon C101 Benchmark (SVG Flat Metric Space)
  console.log('Capturing 09_solomon_svg_benchmark.png');
  await page.click('button[title="Command Center"]');
  await page.waitForTimeout(500);
  await page.selectOption('header select', 'run-solomon-c101-01');
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, '09_solomon_svg_benchmark.png') });

  console.log('All screenshots captured successfully!');

  await browser.close();
  server.kill();
  process.exit(0);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
