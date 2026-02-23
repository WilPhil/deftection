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

        # Create admin user
        User::updateOrCreate(
            [
                'email' => 'admin@example.com'
            ],
            [
                'name' => 'Admin',
                'role' => 'admin',
                'password' => bcrypt('123456'),
            ]
        );

        # Create regular user
        User::updateOrCreate(
            [ 'email' => 'user@example.com' ],
            [
                'name' => 'User',
                'role' => 'user',
                'password' => bcrypt('123456'),
            ]
        );

    }
}
