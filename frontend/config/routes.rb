Rails.application.routes.draw do
  root "predictor#index"

  post "predict", to: "predictor#predict"

  get "analysis", to: "analysis#index"

  get "admin/ingest", to: "admin#ingest", as: :admin_ingest
  post "admin/fetch_schedule", to: "admin#fetch_schedule"
  post "admin/run_ingest", to: "admin#run_ingest"

  get "up" => "rails/health#show", as: :rails_health_check
end
