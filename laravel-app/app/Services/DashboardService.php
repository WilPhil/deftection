<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\RealtimeSession;
use App\Models\RealtimeScan;
use App\Models\ScanDefect;
use App\Models\RealtimeScanDefect;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class DashboardService
{
    /**
     * Get complete dashboard data for the authenticated user
     */
    public function getDashboardData(array $filters = []): array
    {

        $fullTrend = $this->getPerformanceTrend(30);

        $dailyAnalysis = [
            'labels' => array_slice($fullTrend['labels'], -7),
            'totalDefective' => array_slice($fullTrend['data'], -7), // Last 7 days
            'totalProcessed' => array_slice($fullTrend['total_processed'], -7),
        ];

        return [
            'cardData' => $this->getCardDataOnly(),
            'dailyAnalysis' => $dailyAnalysis,
            'defectType' => $this->getDefectTypeDistribution(),
            'performanceTrend' => [
                'labels' => $fullTrend['labels'],
                'defectData' => $fullTrend['data'],
            ],
            'recentAnalyses' => $this->getRecentAnalysesOnly(5),
            'analysesOverview' => $this->getAnalysesOverviewOnly(),
        ];
    }

    /**
     * Get card data for the top 4 statistics cards
     */
    public function getCardDataOnly(): array
    {
        $userId = auth()->id();
        $currentStart = now()->startOfDay();
        $currentEnd = now()->endOfDay();
        $prevStart = now()->subDay()->startOfDay();
        $prevEnd = now()->subDay()->endOfDay();

        // Get standard scan stats for current and previous periods in ONE query
        $scanStats = Scan::where('user_id', $userId)
            ->whereBetween('created_at', [$prevStart, $currentEnd])
            ->selectRaw("
                SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) as current_total,
                SUM(CASE WHEN created_at >= ? AND is_defect = 1 THEN 1 ELSE 0 END) as current_defective,
                SUM(CASE WHEN created_at < ? THEN 1 ELSE 0 END) as prev_total,
                SUM(CASE WHEN created_at < ? AND is_defect = 1 THEN 1 ELSE 0 END) as prev_defect
            ", [$currentStart, $currentStart, $currentStart, $currentStart])
            ->first();

        // Get realtime scan stats for current and previous periods in ONE query
        $realtimeStats = RealtimeScan::join('realtime_sessions', 'realtime_scans.realtime_session_id', '=', 'realtime_sessions.id')
            ->where('realtime_sessions.user_id', $userId)
            ->whereBetween('realtime_scans.created_at', [$prevStart, $currentEnd])
            ->selectRaw("
                SUM(CASE WHEN realtime_scans.created_at >= ? THEN 1 ELSE 0 END) as current_total,
                SUM(CASE WHEN realtime_scans.created_at >= ? AND realtime_scans.is_defect = 1 THEN 1 ELSE 0 END) as current_defective,
                SUM(CASE WHEN realtime_scans.created_at < ? THEN 1 ELSE 0 END) as prev_total,
                SUM(CASE WHEN realtime_scans.created_at < ? AND realtime_scans.is_defect = 1 THEN 1 ELSE 0 END) as prev_defect,
                COUNT(DISTINCT CASE WHEN realtime_scans.created_at >= ? THEN realtime_sessions.id END) as current_sessions,
                COUNT(DISTINCT CASE WHEN realtime_scans.created_at < ? THEN realtime_sessions.id END) as prev_sessions
            ", [$currentStart, $currentStart, $currentStart, $currentStart, $currentStart, $currentStart])
            ->first();

            $currentDefective = ($scanStats->current_defective ?? 0) + ($realtimeStats->current_defective ?? 0);
            $prevDefective = ($scanStats->prev_defect ?? 0) + ($realtimeStats->prev_defect ?? 0);

            $currentScans = $scanStats->current_total ?? 0;
            $prevScans = $scanStats->prev_total ?? 0;

            $currentSessions = $realtimeStats->current_sessions ?? 0;
            $prevSessions = $realtimeStats->prev_sessions ?? 0;

            $currentFrames = $realtimeStats->current_total ?? 0;
            $prevFrames = $realtimeStats->prev_total ?? 0;

        return [
            'totalDefective' => (int)$currentDefective,
            'totalScansImage' => (int)$currentScans,
            'totalRealtimeSessions' => (int)$currentSessions,
            'totalFramesProcessed' => (int)$currentFrames,
            'defectiveChangeRate' => $this->calculateChangeRate((int)$prevDefective, (int)$currentDefective),
            'scansChangeRate' => $this->calculateChangeRate((int)$prevScans, (int)$currentScans),
            'sessionsChangeRate' => $this->calculateChangeRate((int)$prevSessions, (int)$currentSessions),
            'framesChangeRate' => $this->calculateChangeRate((int)$prevFrames, (int)$currentFrames),
        ];
    }

    /**
     * Get defect type distribution
     */
    public function getDefectTypeDistribution(): array
    {
        $userId = auth()->id();

        // Get defects from image scans
        $scanDefects = ScanDefect::join('scans', 'scan_defects.scan_id', '=', 'scans.id')
            ->where('scans.user_id', $userId)
            ->selectRaw('scan_defects.label, COUNT(*) as count')
            ->groupBy('scan_defects.label')
            ->pluck('count', 'label')
            ->toArray();

        // Get defects from realtime scans
        $realtimeDefects = RealtimeScanDefect::join('realtime_scans', 'realtime_scan_defects.realtime_scan_id', '=', 'realtime_scans.id')
            ->join('realtime_sessions', 'realtime_scans.realtime_session_id', '=', 'realtime_sessions.id')
            ->where('realtime_sessions.user_id', $userId)
            ->selectRaw('realtime_scan_defects.label, COUNT(*) as count')
            ->groupBy('realtime_scan_defects.label')
            ->pluck('count', 'label')
            ->toArray();

        // Combine and aggregate the results
        $combined = [];
        foreach(array_unique(array_merge(array_keys($scanDefects), array_keys($realtimeDefects))) as $label) {
            $combined[$label] = ($scanDefects[$label] ?? 0) + ($realtimeDefects[$label] ?? 0);
        }

        // Sort descending so the highest count (Top Defect) is always first
        arsort($combined);

        return [
            'labels' => array_keys($combined),
            'data' => array_values($combined),
        ];
    }

    /**
     * Get performance trend data
     */
    public function getPerformanceTrend(int $days = 30): array
    {
        $userId = auth()->id();
        $startDate = now()->subDays($days - 1)->startOfDay();
        $endDate = now()->endOfDay();

        // Get standard defects by date in ONE query
        $scanStats = Scan::where('user_id', $userId)
            ->where('is_defect', true)
            ->whereBetween('created_at', [$startDate, $endDate])
            ->selectRaw('DATE(created_at) as date, COUNT(*) as total, SUM(is_defect) as defective')
            ->groupBy('date')
            ->get()
            ->keyBy('date');

        // Get realtime defects by date in ONE query
        $realtimeStats = RealtimeScan::join('realtime_sessions', 'realtime_scans.realtime_session_id', '=', 'realtime_sessions.id')
            ->where('realtime_sessions.user_id', $userId)
            ->where('realtime_scans.is_defect', true)
            ->whereBetween('realtime_scans.created_at', [$startDate, $endDate])
            ->selectRaw('DATE(realtime_scans.created_at) as date, COUNT(realtime_scans.id) as total, SUM(realtime_scans.is_defect) as defective')
            ->groupBy('date')
            ->get()
            ->keyBy('date');

        $labels = [];
        $data = [];
        $totalProcessedData = [];

        for ($i = 0; $i < $days; $i++) {
            $dateStr = $startDate->copy()->addDays($i)->format('Y-m-d');
            $labels[] = $dateStr;

            $scanDay = $scanStats->get($dateStr);
            $realtimeDay = $realtimeStats->get($dateStr);

            $defectData[] = (int)($scanDay->defective ?? 0) + (int)($realtimeDay->defective ?? 0);
            $totalProcessedData[] = (int)($scanDay->total ?? 0) + (int)($realtimeDay->total ?? 0);
        }

        return [
            'labels' => $labels,
            'data' => $defectData,
            'total_processed' => $totalProcessedData,
        ];
    }

    /**
     * Get recent analyses
     */
    public function getRecentAnalysesOnly(int $limit = 10): array
    {
        $userId = auth()->id();

        // Get recent scans
        $recentScans = Scan::where('user_id', $userId)
            ->withCount('scanDefects')
            ->orderBy('created_at', 'desc')
            ->limit($limit)
            ->get()
            ->map(function ($scan) {
                return [
                    'id' => $scan->id,
                    'annotated_image_url' => $scan->annotated_image_url,
                    'filename' => $scan->filename,
                    'status' => $scan->is_defect ? 'defect' : 'good',
                    'anomaly_score' => $scan->anomaly_score,
                    'defect_count' => $scan->scan_defects_count,
                    'created_at' => $scan->created_at->format('Y-m-d H:i:s'),
                    'route' => route('scans.show', $scan->id),
                ];
            });

        return $recentScans->toArray();
    }

    /**
     * Calculate change rate between two values
     */
    private function calculateChangeRate(int $previous, int $current): string
    {
        if ($previous == 0) {
            return $current > 0 ? '+100%' : '+0%';
        }

        $change = (($current - $previous) / $previous) * 100;
        $sign = $change >= 0 ? '+' : '';

        return $sign . number_format($change, 1) . '%';
    }

    /**
     * Get analyses overview data (legacy method for backward compatibility)
     */
    public function getAnalysesOverviewOnly(): array
    {
        $userId = auth()->id();

        // Get standard scan stats
        $scanStats = Scan::where('user_id', $userId)
            ->selectRaw("
                COUNT(id) as total_processed,
                SUM(CASE WHEN is_defect = 1 THEN 1 ELSE 0 END) as total_defective,
                AVG(COALESCE(preprocessing_time_ms, 0) + COALESCE(anomaly_inference_time_ms, 0) + COALESCE(classification_inference_time_ms, 0) + COALESCE(postprocessing_time_ms, 0)) as avg_processing_time
            ")
            ->first();

        // Get realtime scan stats
        $realtimeStats = RealtimeScan::join('realtime_sessions', 'realtime_scans.realtime_session_id', '=', 'realtime_sessions.id')
            ->where('realtime_sessions.user_id', $userId)
            ->selectRaw("
                COUNT(realtime_scans.id) as total_processed,
                SUM(CASE WHEN realtime_scans.is_defect = 1 THEN 1 ELSE 0 END) as total_defective,
                AVG(COALESCE(realtime_scans.preprocessing_time_ms, 0) + COALESCE(realtime_scans.anomaly_inference_time_ms, 0) + COALESCE(realtime_scans.classification_inference_time_ms, 0) + COALESCE(realtime_scans.postprocessing_time_ms, 0)) as avg_processing_time
            ")
            ->first();

        // Get average AI confidence
        $scanConfidenceAvg = ScanDefect::join('scans', 'scan_defects.scan_id', '=', 'scans.id')
            ->where('scans.user_id', $userId)
            ->avg('scan_defects.confidence_score') ?? 0;

        $realtimeConfidenceAvg = RealtimeScanDefect::join('realtime_scans', 'realtime_scan_defects.realtime_scan_id', '=', 'realtime_scans.id')
            ->join('realtime_sessions', 'realtime_scans.realtime_session_id', '=', 'realtime_sessions.id')
            ->where('realtime_sessions.user_id', $userId)
            ->avg('realtime_scan_defects.confidence_score') ?? 0;

        // Math
        $totalProcessed = ($scanStats->total_processed ?? 0) + ($realtimeStats->total_processed ?? 0);
        $totalDefective = ($scanStats->total_defective ?? 0) + ($realtimeStats->total_defective ?? 0);
        $totalSuccess = $totalProcessed - $totalDefective;

        // Calculate rates
        $goodRate = $totalProcessed > 0 ? round(($totalSuccess / $totalProcessed) * 100, 2) : 0;
        $defectRate = $totalProcessed > 0 ? round(($totalDefective / $totalProcessed) * 100, 2) : 0;

        // Calculate average processing time
        $validTimes = array_filter([$scanStats->avg_processing_time ?? 0, $realtimeStats->avg_processing_time ?? 0]);
        $avgProcessingTime = empty($validTimes) ? 0 : round(array_sum($validTimes) / count($validTimes), 2);

        // Calculate average AI confidence
        $validConfidences = array_filter([$scanConfidenceAvg, $realtimeConfidenceAvg]);
        $avgAiConfidence = empty($validConfidences) ? 0.0 : round((array_sum($validConfidences) / count($validConfidences)) * 100, 2);

        return [
            'avgProcessingTime' => $avgProcessingTime,
            'goodRate' => $goodRate,
            'defectRate' => $defectRate,
            'aiConfidence' => $avgAiConfidence,
        ];
    }
}
