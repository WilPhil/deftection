<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\Scan;
use Illuminate\Http\Request;
use App\Models\RealtimeSession;
use App\Services\ReportService;
use Illuminate\Support\Facades\Log;
use Barryvdh\Snappy\Facades\SnappyPdf;
use App\Services\RealtimeSessionReportService;
use Illuminate\Validation\ValidationException;
use Illuminate\Auth\Access\AuthorizationException;

class ReportController extends Controller
{

    public function __construct(protected ReportService $reportService, protected RealtimeSessionReportService $realtimeReportService) {}


    /**
     * Generate realtime session report
     */
    public function generateRealtimeSessionReport(Request $request, RealtimeSession $realtimeSession)
    {
        try {
            $this->authorize('generateReport', $realtimeSession);

            // Get report data
            $reportData = $this->realtimeReportService->prepareRealtimeSessionReport($realtimeSession);

            // Generate PDF with Snappy
            $pdf = SnappyPdf::loadView('reports.realtime-session', $reportData)
                ->setOptions([
                    'page-size' => 'A4',
                    'orientation' => 'Portrait',
                    'margin-top' => '15mm',
                    'margin-right' => '10mm',
                    'margin-bottom' => '15mm',
                    'margin-left' => '10mm',
                    'encoding' => 'UTF-8',
                    'enable-local-file-access' => true,
                    'footer-center' => 'Page [page] of [topage]',
                    'footer-font-size' => 8,
                ]);

            $filename = "realtime-session-report-{$realtimeSession->id}-" . now()->format('Y-m-d-H-i-s') . ".pdf";

            Log::info('Realtime session report generated successfully', [
                'session_id' => $realtimeSession->id,
                'user_id' => auth()->id(),
                'filename' => $filename,
                'total_frames' => $realtimeSession->total_frames_processed
            ]);

            return $pdf->download($filename);
        } catch (AuthorizationException $e) {
            Log::warning('Unauthorized realtime session report access attempt', [
                'session_id' => $realtimeSession->id,
                'user_id' => auth()->id(),
                'error' => $e->getMessage()
            ]);
            return back()->with('error', 'You are not authorized to generate this report.');
        } catch (\Exception $e) {
            Log::error('Realtime session report generation failed', [
                'session_id' => $realtimeSession->id,
                'user_id' => auth()->id(),
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString()
            ]);

            return back()->with('error', 'Failed to generate session report. Please try again.');
        }
    }

    /**
     * Generate single scan report
     */
    public function generateSingleReport(Request $request, Scan $scan)
    {
        try {
            $this->authorize('generateReport', $scan);

            // Get report data
            $reportData = $this->reportService->prepareSingleScanReport($scan);

            // Generate PDF with Snappy
            $pdf = SnappyPdf::loadView('reports.single-scan', $reportData)
                ->setOptions([
                    'page-size' => 'A4',
                    'orientation' => 'Portrait',
                    'margin-top' => '20mm',
                    'margin-right' => '15mm',
                    'margin-bottom' => '20mm',
                    'margin-left' => '15mm',
                    'encoding' => 'UTF-8',
                    'enable-local-file-access' => true,
                ]);

            $filename = "scan-report-{$scan->id}-" . now()->format('Y-m-d-H-i-s') . ".pdf";

            Log::info('Single scan report generated successfully', [
                'scan_id' => $scan->id,
                'user_id' => auth()->id(),
                'filename' => $filename
            ]);

            return $pdf->download($filename);
        } catch (\Exception $e) {
            // Log::error('Single scan report generation failed', [
            //     'scan_id' => $scan->id,
            //     'user_id' => auth()->id(),
            //     'error' => $e->getMessage(),
            // ]);

            // return back()->with('error', 'Failed to generate report. Please try again.');
            dd($e->getMessage());
        }
    }

    /**
     * Generate batch scan report
     */
    public function generateBatchReport(Request $request)
    {
        try {
            $this->authorize('generateBatchReport', Scan::class);

            $request->validate([
                'dateFrom' => 'nullable|date',
                'dateTo' => 'nullable|date|after_or_equal:dateFrom',
                'defectTypes' => 'nullable|array',
                'defectTypes.*' => 'string',
                'userId' => 'nullable|integer|exists:users,id',
                'users' => 'nullable|array',
                'users.*' => 'string', // string because frontend sends string IDs
                'roles' => 'nullable|array',
                'roles.*' => 'string|in:admin,user',
                'defect_types' => 'nullable|array',
                'defect_types.*' => 'string',
                'date_from' => 'nullable|date',
                'date_to' => 'nullable|date|after_or_equal:date_from',
                'sort_by' => 'nullable|string',
                'sort_dir' => 'nullable|string|in:asc,desc',
            ]);

            // Handle status field, it can be array or string
            $statusInput = $request->input('status');
            if (is_array($statusInput)) {
                $status = $statusInput[0] ?? 'all';
            } else {
                $status = $statusInput ?? 'all';
            }

            $statusMapping = [
                'defect' => 'defective',
                'defective' => 'defective',
                'good' => 'good',
                'all' => 'all'
            ];

            $normalizedStatus = $statusMapping[$status] ?? 'all';

            // Normalize (handle both snake_case and camelCase)
            $filters = [
                'dateFrom' => $request->input('dateFrom') ?? $request->input('date_from'),
                'dateTo' => $request->input('dateTo') ?? $request->input('date_to'),
                'defectTypes' => $request->input('defectTypes') ?? $request->input('defect_types') ?? [],
                'status' => $normalizedStatus,
                'users' => [],
                'roles' => $request->input('roles') ?? [],
            ];

            // Handle users array - convert string ID to integer
            $usersInput = $request->input('users', []);
            if (!empty($usersInput)) {
                $filters['users'] = array_map('intval', array_filter($usersInput));
            }

            // Set default date range if not provided (last 30 days)
            if (!$filters['dateFrom'] && !$filters['dateTo']) {
                $filters['dateTo'] = Carbon::now();
                $filters['dateFrom'] = Carbon::now()->subDays(30);
            } elseif ($filters['dateFrom'] && !$filters['dateTo']) {
                $filters['dateTo'] = Carbon::parse($filters['dateFrom'])->addDays(30);
            } elseif (!$filters['dateFrom'] && $filters['dateTo']) {
                $filters['dateFrom'] = Carbon::parse($filters['dateTo'])->subDays(30);
            } else {
                $filters['dateFrom'] = Carbon::parse($filters['dateFrom']);
                $filters['dateTo'] = Carbon::parse($filters['dateTo']);
            }

            // Set time boundaries
            $filters['dateFrom'] = $filters['dateFrom']->startOfDay();
            $filters['dateTo'] = $filters['dateTo']->endOfDay();

            // Apply authorization-based user filtering
            if (auth()->user()->can('generateAnyReport', Scan::class)) {
                // Admin can generate reports for specific users or all users
                // If no users specified, leave empty array (will show all users)
                $filters['users'] = $filters['users']; // Keep as provided
            } else {
                // Regular users can only generate reports for themselves
                $filters['users'] = [auth()->id()];
            }

            // Check date range (limit to prevent server overload)
            $daysDiff = $filters['dateFrom']->diffInDays($filters['dateTo']);
            if ($daysDiff > 365) {
                return back()->with('error', 'Date range cannot exceed 365 days.');
            }

            Log::info('Batch report generation started', [
                'user_id' => auth()->id(),
                'filters' => $filters,
                'original_status_input' => $statusInput,
                'normalized_status' => $normalizedStatus,
                'user_can_generate_any' => auth()->user()->can('generateAnyReport', Scan::class)
            ]);

            // Get report data
            $reportData = $this->reportService->prepareBatchScanReport($filters);

            // Check if any data found
            if (empty($reportData['scans']) || $reportData['summary']['total_scans'] === 0) {
                Log::warning('No data found for batch report', [
                    'user_id' => auth()->id(),
                    'filters' => $filters
                ]);
                return back()->with('error', 'No scan data found for the specified criteria.');
            }

            // Generate PDF with Snappy
            $pdf = SnappyPdf::loadView('reports.batch-scan', $reportData)
                ->setOptions([
                    'page-size' => 'A4',
                    'orientation' => 'Portrait',
                    'margin-top' => '15mm',
                    'margin-right' => '10mm',
                    'margin-bottom' => '15mm',
                    'margin-left' => '10mm',
                    'encoding' => 'UTF-8',
                    'enable-local-file-access' => true,
                    'footer-center' => 'Page [page] of [topage]',
                    'footer-font-size' => 8,
                ]);

            $filename = "batch-scan-report-" . $filters['dateFrom']->format('Y-m-d') . "-to-" . $filters['dateTo']->format('Y-m-d') . "-" . now()->format('H-i-s') . ".pdf";

            Log::info('Batch scan report generated successfully', [
                'user_id' => auth()->id(),
                'filename' => $filename,
                'total_scans' => $reportData['summary']['total_scans']
            ]);

            return $pdf->download($filename);
        } catch (ValidationException $e) {
            Log::warning('Batch report validation failed', [
                'user_id' => auth()->id(),
                'errors' => $e->errors(),
                'input' => $request->all()
            ]);
            return back()->withErrors($e->errors())->withInput();
        } catch (AuthorizationException $e) {
            Log::warning('Unauthorized batch report access attempt', [
                'user_id' => auth()->id(),
                'error' => $e->getMessage()
            ]);
            return back()->with('error', 'You are not authorized to generate this report.');
        } catch (\Exception $e) {
            Log::error('Batch scan report generation failed', [
                'user_id' => auth()->id(),
                'filters' => $filters ?? [],
                'request_data' => $request->all(),
                'error' => $e->getMessage(),
            ]);

            return back()->with('error', 'Failed to generate batch report. Please try again.');
        }
    }

    /**
     * Preview batch scan report (for testing)
     */
    public function previewBatchReport(Request $request)
    {
        try {
            $this->authorize('generateBatchReport', Scan::class);

            //  same validation as generateBatchReport
            $request->validate([
                'dateFrom' => 'required|date',
                'dateTo' => 'required|date|after_or_equal:dateFrom',
                'defectTypes' => 'nullable|array',
                'defectTypes.*' => 'string',
                'userId' => 'nullable|integer|exists:users,id',
                'users' => 'nullable|array',
                'users.*' => 'string',
            ]);

            // Handle status field
            $statusInput = $request->input('status');
            if (is_array($statusInput)) {
                $status = $statusInput[0] ?? 'all';
            } else {
                $status = $statusInput ?? 'all';
            }

            $statusMapping = [
                'defect' => 'defective',
                'defective' => 'defective',
                'good' => 'good',
                'all' => 'all'
            ];

            $normalizedStatus = $statusMapping[$status] ?? 'all';

            // Parse dates
            $dateFrom = Carbon::parse($request->input('dateFrom'))->startOfDay();
            $dateTo = Carbon::parse($request->input('dateTo'))->endOfDay();

            // Set user filter with authorization
            $usersInput = $request->input('users', []);
            $users = [];
            if (auth()->user()->can('generateAnyReport', Scan::class)) {
                if (!empty($usersInput)) {
                    $users = array_map('intval', array_filter($usersInput));
                } elseif ($request->input('userId')) {
                    $users = [$request->input('userId')];
                }
            } else {
                $users = [auth()->id()];
            }

            // Prepare filter parameters
            $filters = [
                'dateFrom' => $dateFrom,
                'dateTo' => $dateTo,
                'defectTypes' => $request->input('defectTypes') ?? [],
                'status' => $normalizedStatus,
                'users' => $users,
                'roles' => $request->input('roles') ?? [],
            ];

            // Get report data
            $reportData = $this->reportService->prepareBatchScanReport($filters);

            return view('reports.batch-scan', $reportData);
        } catch (\Exception $e) {
            Log::error('Batch scan report preview failed', [
                'user_id' => auth()->id(),
                'error' => $e->getMessage()
            ]);

            return back()->with('error', 'Failed to preview batch report. Please try again.');
        }
    }

    /**
     * Preview single scan report (for testing)
     */
    public function previewSingleReport(Scan $scan)
    {
        try {
            // Authorization check using policy
            $this->authorize('generateReport', $scan);

            // Get report data
            $reportData = $this->reportService->prepareSingleScanReport($scan);

            return view('reports.single-scan', $reportData);
        } catch (\Exception $e) {
            Log::error('Single scan report preview failed', [
                'scan_id' => $scan->id,
                'user_id' => auth()->id(),
                'error' => $e->getMessage()
            ]);

            return back()->with('error', 'Failed to preview report. Please try again.');
        }
    }
}
