<?php

namespace Database\Seeders;

// use Illuminate\Database\Console\Seeds\WithoutModelEvents;

use App\Models\User;
use App\Models\DefectType;
use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     */
    public function run(): void
    {
        $this->call([
            // ProductSeeder::class,
            DefectTypeSeeder::class,
            // MockSeeder::class,


        ]);

        User::factory(3)->updateOrCreate()->each(
            function ($user, $index) {
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
                if ($index == 2) {
                    $user->update([
                        'name' => 'admin ' . $index,
                        'role' => 'admin',
                        'email' => 'admin2@example.com',
                        'password' => bcrypt('123456'),
                    ]);
                }
            }
        );
    }
}
