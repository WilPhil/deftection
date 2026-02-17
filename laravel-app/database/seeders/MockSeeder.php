<?php

namespace Database\Seeders;

use App\Models\Scan;
use App\Models\ScanDefect;
use App\Models\ScanThreat;
use App\Models\User;
use App\Models\RealtimeSession;
use App\Models\RealtimeScan;
use App\Models\RealtimeScanDefect;
use Illuminate\Database\Seeder;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;

class MockSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        User::factory(6)->updateOrCreate()->each(function ($user, $index) {
            if ($index == 0) {
                $user->update([
                    'name' => 'admin ' . $index,
                    'role' => 'admin',
                    'email' => 'admin@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }
            if ($index == 1) {
                $user->update([
                    'name' => 'user ' . $index,
                    'role' => 'user',
                    'email' => 'user@example.com',
                    'password' => bcrypt('123456'),
                ]);
            }

            // Create regular scans
            Scan::factory(25)->updateOrCreate([
                'user_id' => $user->id,
            ])->each(function ($scan) {
                $createdAt = fake()->dateTimeBetween('-1 month', 'now');

                $scan->update([
                    'created_at' => $createdAt,
                    'updated_at' => $createdAt,
                ]);

                if ($scan->is_defect) {
                    $rand1 = rand(1, 5);

                    ScanDefect::factory($rand1)->updateOrCreate([
                        'scan_id' => $scan->id,
                        'created_at' => $createdAt,
                        'updated_at' => $createdAt
                    ]);
                }

                $rand2 = rand(0, 2);
                if ($rand2 == 0) {
                    ScanThreat::factory()->updateOrCreate([
                        'scan_id' => $scan->id,
                        'status' => 'clean',
                        'risk_level' => 'none',
                        'flags' => null,
                        'details' => null,
                        'possible_attack' => null,
                        'created_at' => $createdAt,
                        'updated_at' => $createdAt
                    ]);
                } elseif ($rand2 == 1) {
                    ScanThreat::factory()->updateOrCreate([
                        'scan_id' => $scan->id,
                        'created_at' => $createdAt,
                        'updated_at' => $createdAt
                    ]);
                }
            });

            // Create realtime sessions for each user
            RealtimeSession::factory(rand(3, 8))->updateOrCreate([
                'user_id' => $user->id,
            ])->each(function ($session) {
                $sessionCreatedAt = fake()->dateTimeBetween('-1 month', 'now');

                // Ensure session end time is properly set based on status
                $sessionEndTime = $session->session_end;
                if ($session->session_status === 'completed' && !$sessionEndTime) {
                    $sessionEndTime = fake()->dateTimeBetween($sessionCreatedAt, 'now');
                } elseif ($session->session_status === 'aborted' && !$sessionEndTime) {
                    $maxEnd = clone $sessionCreatedAt;
                    $maxEnd->modify('+1 hour');
                    $sessionEndTime = fake()->dateTimeBetween($sessionCreatedAt, min(new \DateTime(), $maxEnd));
                }

                $session->update([
                    'session_start' => $sessionCreatedAt,
                    'session_end' => $sessionEndTime,
                    'created_at' => $sessionCreatedAt,
                    'updated_at' => $sessionEndTime ?? $sessionCreatedAt,
                ]);

                // Create realtime scans for each session
                $scanCount = rand(50, 200); // Random number of scans per session

                RealtimeScan::factory($scanCount)->updateOrCreate([
                    'realtime_session_id' => $session->id,
                ])->each(function ($realtimeScan, $scanIndex) use ($session, $sessionCreatedAt) {
                    // Generate captured_at time within session duration
                    $capturedAt = $sessionCreatedAt;

                    if ($session->session_end) {
                        // Only generate time between start and end if end time is after start time
                        if ($session->session_end > $sessionCreatedAt) {
                            $capturedAt = fake()->dateTimeBetween($sessionCreatedAt, $session->session_end);
                        }
                    } else {
                        // For ongoing sessions, generate times within reasonable range
                        $maxTime = clone $sessionCreatedAt;
                        $maxTime->modify('+2 hours');
                        $endTime = min(new \DateTime(), $maxTime);

                        if ($endTime > $sessionCreatedAt) {
                            $capturedAt = fake()->dateTimeBetween($sessionCreatedAt, $endTime);
                        }
                    }

                    $realtimeScan->update([
                        'captured_at' => $capturedAt,
                        'created_at' => $capturedAt,
                        'updated_at' => $capturedAt,
                    ]);

                    // Create defects for scans that are marked as defective
                    if ($realtimeScan->is_defect) {
                        $defectCount = rand(1, 3); // 1-3 defects per defective scan

                        RealtimeScanDefect::factory($defectCount)->updateOrCreate([
                            'realtime_scan_id' => $realtimeScan->id,
                            'created_at' => $capturedAt,
                            'updated_at' => $capturedAt,
                        ]);
                    }
                });

                // Update session statistics based on created scans
                $this->updateSessionStatistics($session);
            });
        });
    }

    /**
     * Update realtime session statistics based on its scans
     */
    private function updateSessionStatistics(RealtimeSession $session): void
    {
        $scans = $session->realtimeScans;
        $totalScans = $scans->count();

        if ($totalScans === 0) {
            return;
        }

        $defectScans = $scans->where('is_defect', true);
        $defectCount = $defectScans->count();
        $goodCount = $totalScans - $defectCount;

        $defectRate = $totalScans > 0 ? round(($defectCount / $totalScans) * 100, 2) : 0;
        $goodRate = $totalScans > 0 ? round(($goodCount / $totalScans) * 100, 2) : 0;

        // Calculate processing times
        $processingTimes = $scans->pluck('anomaly_inference_time_ms')->filter();
        $anomalyScores = $scans->pluck('anomaly_score')->filter();

        // Get defect type distribution
        $defectTypes = $defectScans->flatMap(function ($scan) {
            return $scan->realtimeScanDefects->pluck('label');
        })->countBy()->toArray();

        // Get severity distribution
        $severityLevels = $defectScans->flatMap(function ($scan) {
            return $scan->realtimeScanDefects->pluck('severity_level');
        })->countBy()->toArray();

        $session->update([
            'total_frames_processed' => $totalScans,
            'defect_count' => $defectCount,
            'defect_rate' => $defectRate,
            'good_count' => $goodCount,
            'good_rate' => $goodRate,
            'avg_processing_time' => $processingTimes->avg(),
            'max_processing_time' => $processingTimes->max(),
            'min_processing_time' => $processingTimes->min(),
            'avg_anomaly_score' => $anomalyScores->avg(),
            'max_anomaly_score' => $anomalyScores->max(),
            'min_anomaly_score' => $anomalyScores->min(),
            'defect_type_distribution' => $defectTypes,
            'severity_level_distribution' => $severityLevels,
        ]);
    }
}
