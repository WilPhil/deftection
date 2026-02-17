<?php

namespace Database\Seeders;

use App\Models\DefectType;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class DefectTypeSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        //
        // DefectType::factory(5)->create();

        $defectTypes = [
            [
                'id' => 1,
                'name' => 'damaged',
                'slug' => 'damaged',
                'description' => 'Item is broken, cracked, dented, or does not work properly',
            ],
            [
                'id' => 2,
                'name' => 'Missing Component',
                'slug' => 'missing_component',
                'description' => 'Item is missing parts or pieces that should be included',
            ],
            [
                'id' => 3,
                'name' => 'open',
                'slug' => 'open',
                'description' => 'Package has been opened or box is not sealed',
            ],
            [
                'id' => 4,
                'name' => 'scratch',
                'slug' => 'scratch',
                'description' => 'Item has scratches or marks on the surface',
            ],
            [
                'id' => 5,
                'name' => 'stained',
                'slug' => 'stained',
                'description' => 'Item has stains, spots, or dirty marks on it',
            ],
        ];

        foreach ($defectTypes as $defectType) {
            DefectType::updateOrCreate($defectType);
        }
    }
}
