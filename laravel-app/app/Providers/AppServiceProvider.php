<?php

namespace App\Providers;

use App\Models\Scan;
use App\Policies\ScanPolicy;
use Illuminate\Support\Facades\Gate;
use Illuminate\Support\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        //
        Gate::policy(Scan::class, ScanPolicy::class);

        if (app()->environment('production')) {
            \URL::forcheScheme('https');
        }
    }
}
