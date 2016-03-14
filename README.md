# Help.jl

Search through readme and documentation of packages and functions based on a query.  Unlike `apropos`, Help uses TF-IDF instead of sub-string search.

Run `update()` first to build the data needed for search.

## Example

```julia
julia> using Help
julia> help("Gray scale an image")
Help Results
---------
1. Images
2. ImageView
3. TestImages
4. ImageRegistration
5. SloanDigitalSkySurvey

julia> using MySQL
julia> help(MySQL, "get the last inserted id")
Help Results
---------
1. mysql_insert_id

julia> help(Base, "convert pointer to array")
Help Results
---------
1. pointer_to_array
2. oftype
3. bitpack
```
