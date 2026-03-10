require "httparty"

class F1ApiClient
  include HTTParty

  base_uri ENV.fetch("F1_API_URL", "http://localhost:8000")

  def self.drivers
    response = get("/api/drivers")
    handle_response(response)["drivers"]
  rescue StandardError => e
    Rails.logger.error("F1 API error (drivers): #{e.message}")
    []
  end

  def self.circuits
    response = get("/api/circuits")
    handle_response(response)["circuits"]
  rescue StandardError => e
    Rails.logger.error("F1 API error (circuits): #{e.message}")
    []
  end

  def self.predict(driver:, circuit:, grid:)
    response = post("/api/predict",
      body: { driver: driver, circuit: circuit, grid: grid }.to_json,
      headers: { "Content-Type" => "application/json" }
    )
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (predict): #{e.message}")
    nil
  end

  def self.single_race
    response = get("/api/analysis/single-race")
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (single-race): #{e.message}")
    nil
  end

  def self.stats
    response = get("/api/analysis/stats")
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (stats): #{e.message}")
    nil
  end

  def self.confusion_matrix
    response = get("/api/analysis/confusion-matrix")
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (confusion-matrix): #{e.message}")
    nil
  end

  def self.ingest(year:, round:)
    response = post("/api/data/ingest",
      body: { year: year, round: round }.to_json,
      headers: { "Content-Type" => "application/json" }
    )
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (ingest): #{e.message}")
    nil
  end

  def self.schedule(year:)
    response = get("/api/schedule/#{year}")
    handle_response(response)
  rescue StandardError => e
    Rails.logger.error("F1 API error (schedule): #{e.message}")
    nil
  end

  private

  def self.handle_response(response)
    if response.success?
      response.parsed_response
    else
      raise "API returned #{response.code}: #{response.body}"
    end
  end
end
