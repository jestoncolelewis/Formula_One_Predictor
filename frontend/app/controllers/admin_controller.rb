class AdminController < ApplicationController
  def ingest
    @schedule = nil
  end

  def fetch_schedule
    year = params[:year].to_i
    result = F1ApiClient.schedule(year: year)
    if result
      @schedule = result["races"]
      @year = year
    else
      flash.now[:alert] = "Failed to fetch schedule."
    end
    render :ingest
  end

  def run_ingest
    year = params[:year].to_i
    round = params[:round].to_i

    result = F1ApiClient.ingest(year: year, round: round)
    if result
      flash[:notice] = "#{result['message']}. Model accuracy: #{result['model_accuracy']}%"
    else
      flash[:alert] = "Failed to ingest race data. Check the API logs."
    end
    redirect_to admin_ingest_path
  end
end
