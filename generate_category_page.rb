require 'yaml'

# navigation.yml 파일 읽기
nav_data = YAML.load_file('_data/navigation.yml')

# _pages 디렉토리가 없으면 생성
Dir.mkdir('_pages') unless Dir.exist?('_pages')

nav_data['docs'].each do |section|
  section['children'].each do |item|
    if item['category']
      # 각 카테고리에 대한 마크다운 파일 생성
      File.open("_pages/category/#{item['category']}.md", 'w') do |file|
        file.puts "---"
        file.puts "layout: category"
        file.puts "title: \"#{item['title']}\""
        file.puts "taxonomy: #{item['category']}"
        file.puts "permalink: #{item['url']}"
        file.puts "author_profile: false"
        file.puts "sidebar:"
        file.puts "  nav: \"docs\""
        file.puts "---"
      end
    end
  end
end

puts "카테고리 페이지가 생성되었습니다."